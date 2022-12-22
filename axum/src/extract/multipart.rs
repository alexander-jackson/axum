//! Extractor that parses `multipart/form-data` requests commonly used with file uploads.
//!
//! See [`Multipart`] for more details.

use super::{BodyStream, FromRequest};
use crate::body::{Bytes, HttpBody};
use crate::BoxError;
use async_trait::async_trait;
use axum_core::RequestExt;
use futures_util::stream::{Stream, StreamExt};
use http::header::{HeaderMap, CONTENT_TYPE};
use http::Request;
use std::{
    fmt,
    pin::Pin,
    task::{Context, Poll},
};

/// Extractor that parses `multipart/form-data` requests (commonly used with file uploads).
///
/// Since extracting multipart form data from the request requires consuming the body, the
/// `Multipart` extractor must be *last* if there are multiple extractors in a handler.
/// See ["the order of extractors"][order-of-extractors]
///
/// [order-of-extractors]: crate::extract#the-order-of-extractors
///
/// # Example
///
/// ```rust,no_run
/// use axum::{
///     extract::Multipart,
///     routing::post,
///     Router,
/// };
/// use futures::stream::StreamExt;
///
/// async fn upload(mut multipart: Multipart) {
///     while let Some(mut field) = multipart.next_field().await.unwrap() {
///         let name = field.name().unwrap().to_string();
///         let data = field.bytes().await.unwrap();
///
///         println!("Length of `{}` is {} bytes", name, data.len());
///     }
/// }
///
/// let app = Router::new().route("/upload", post(upload));
/// # async {
/// # axum::Server::bind(&"".parse().unwrap()).serve(app.into_make_service()).await.unwrap();
/// # };
/// ```
#[cfg_attr(docsrs, doc(cfg(feature = "multipart")))]
#[derive(Debug)]
pub struct Multipart {
    inner: multer::Multipart<'static>,
}

#[async_trait]
impl<S, B> FromRequest<S, B> for Multipart
where
    B: HttpBody + Send + 'static,
    B::Data: Into<Bytes>,
    B::Error: Into<BoxError>,
    S: Send + Sync,
{
    type Rejection = MultipartRejection;

    async fn from_request(req: Request<B>, state: &S) -> Result<Self, Self::Rejection> {
        let boundary = parse_boundary(req.headers()).ok_or(InvalidBoundary)?;
        let stream_result = match req.with_limited_body() {
            Ok(limited) => BodyStream::from_request(limited, state).await,
            Err(unlimited) => BodyStream::from_request(unlimited, state).await,
        };
        let stream = stream_result.unwrap_or_else(|err| match err {});
        let multipart = multer::Multipart::new(stream, boundary);
        Ok(Self { inner: multipart })
    }
}

impl Multipart {
    /// Yields the next [`Field`] if available.
    pub async fn next_field(&mut self) -> Result<Option<Field<'_>>, MultipartError> {
        let field = self
            .inner
            .next_field()
            .await
            .map_err(MultipartError::from_multer)?;

        if let Some(field) = field {
            Ok(Some(Field {
                inner: field,
                _multipart: self,
            }))
        } else {
            Ok(None)
        }
    }

    /// Convert the multipart body into a stream by mapping each field.
    ///
    /// This enables processing each field in a stream, without buffering the contents all at
    /// once.
    ///
    /// # Example
    ///
    /// Upload each part without buffering the contents of each field:
    ///
    /// ```
    /// use futures::prelude::*;
    /// use axum::{
    ///     Router,
    ///     routing::post,
    ///     extract::Multipart,
    ///     BoxError,
    ///     http::StatusCode,
    ///     body::Bytes,
    /// };
    ///
    /// async fn handler(multipart: Multipart) -> Result<(), StatusCode> {
    ///     let stream = multipart
    ///         // convert into a stream of file name + stream of contents
    ///         .into_stream(|field| {
    ///             if let Some(file_name) = field.file_name() {
    ///                 Some((file_name.to_owned(), field.into_stream()))
    ///             } else {
    ///                 None
    ///             }
    ///         })
    ///         // ignore fields that don't have a file name
    ///         .try_filter_map(|item| std::future::ready(Ok(item)));
    ///
    ///     // pin the stream to the stack. This is necessary to consume the stream
    ///     futures::pin_mut!(stream);
    ///
    ///     // consume the stream and upload each file
    ///     while let Some(Ok((file_name, contents))) = stream.next().await {
    ///         upload_file(file_name, contents)
    ///             .await
    ///             .map_err(|err| StatusCode::BAD_REQUEST)?;
    ///     }
    ///
    ///     Ok(())
    /// }
    ///
    /// async fn upload_file<S, E>(file_name: String, contents: S) -> Result<(), E>
    /// where
    ///     S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    /// {
    ///     // ...
    ///     # Ok(())
    /// }
    ///
    /// let app = Router::new().route("/upload", post(handler));
    /// # let _: Router = app;
    /// ```
    pub fn into_stream<F, T>(
        self,
        map_field: F,
    ) -> impl Stream<Item = Result<T, MultipartError>> + Send + 'static
    where
        F: FnMut(Field<'_>) -> T + Clone + Send + 'static,
    {
        futures_util::stream::try_unfold(self, move |mut multipart| {
            let mut map_field = map_field.clone();
            async move {
                match multipart.next_field().await? {
                    Some(field) => Ok(Some((map_field(field), multipart))),
                    None => Ok(None),
                }
            }
        })
    }
}

/// A single field in a multipart stream.
#[derive(Debug)]
pub struct Field<'a> {
    inner: multer::Field<'static>,
    // multer requires there to only be one live `multer::Field` at any point. This enforces that
    // statically, which multer does not do, it returns an error instead.
    _multipart: &'a mut Multipart,
}

impl<'a> Stream for Field<'a> {
    type Item = Result<Bytes, MultipartError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner)
            .poll_next(cx)
            .map_err(MultipartError::from_multer)
    }
}

impl<'a> Field<'a> {
    /// The field name found in the
    /// [`Content-Disposition`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Disposition)
    /// header.
    pub fn name(&self) -> Option<&str> {
        self.inner.name()
    }

    /// The file name found in the
    /// [`Content-Disposition`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Disposition)
    /// header.
    pub fn file_name(&self) -> Option<&str> {
        self.inner.file_name()
    }

    /// Get the [content type](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Type) of the field.
    pub fn content_type(&self) -> Option<&str> {
        self.inner.content_type().map(|m| m.as_ref())
    }

    /// Get a map of headers as [`HeaderMap`].
    pub fn headers(&self) -> &HeaderMap {
        self.inner.headers()
    }

    /// Get the full data of the field as [`Bytes`].
    pub async fn bytes(self) -> Result<Bytes, MultipartError> {
        self.inner
            .bytes()
            .await
            .map_err(MultipartError::from_multer)
    }

    /// Get the full field data as text.
    pub async fn text(self) -> Result<String, MultipartError> {
        self.inner.text().await.map_err(MultipartError::from_multer)
    }

    /// Stream a chunk of the field data.
    ///
    /// When the field data has been exhausted, this will return [`None`].
    ///
    /// Note this does the same thing as `Field`'s [`Stream`] implementation.
    ///
    /// # Example
    ///
    /// ```
    /// use axum::{
    ///    extract::Multipart,
    ///    routing::post,
    ///    response::IntoResponse,
    ///    http::StatusCode,
    ///    Router,
    /// };
    ///
    /// async fn upload(mut multipart: Multipart) -> Result<(), (StatusCode, String)> {
    ///     while let Some(mut field) = multipart
    ///         .next_field()
    ///         .await
    ///         .map_err(|err| (StatusCode::BAD_REQUEST, err.to_string()))?
    ///     {
    ///         while let Some(chunk) = field
    ///             .chunk()
    ///             .await
    ///             .map_err(|err| (StatusCode::BAD_REQUEST, err.to_string()))?
    ///         {
    ///             println!("received {} bytes", chunk.len());
    ///         }
    ///     }
    ///
    ///     Ok(())
    /// }
    ///
    /// let app = Router::new().route("/upload", post(upload));
    /// # let _: Router = app;
    /// ```
    pub async fn chunk(&mut self) -> Result<Option<Bytes>, MultipartError> {
        self.inner
            .chunk()
            .await
            .map_err(MultipartError::from_multer)
    }

    /// Convert the field into a stream of chunks.
    pub fn into_stream(self) -> impl Stream<Item = Result<Bytes, MultipartError>> + Send + 'static {
        let field = self.inner;

        futures_util::stream::try_unfold(field, move |mut field| async move {
            if let Some(chunk) = field
                .next()
                .await
                .transpose()
                .map_err(MultipartError::from_multer)?
            {
                Ok(Some((chunk, field)))
            } else {
                Ok(None)
            }
        })
    }
}

/// Errors associated with parsing `multipart/form-data` requests.
#[derive(Debug)]
pub struct MultipartError {
    source: multer::Error,
}

impl MultipartError {
    fn from_multer(multer: multer::Error) -> Self {
        Self { source: multer }
    }
}

impl fmt::Display for MultipartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Error parsing `multipart/form-data` request")
    }
}

impl std::error::Error for MultipartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

fn parse_boundary(headers: &HeaderMap) -> Option<String> {
    let content_type = headers.get(CONTENT_TYPE)?.to_str().ok()?;
    multer::parse_boundary(content_type).ok()
}

composite_rejection! {
    /// Rejection used for [`Multipart`].
    ///
    /// Contains one variant for each way the [`Multipart`] extractor can fail.
    pub enum MultipartRejection {
        InvalidBoundary,
    }
}

define_rejection! {
    #[status = BAD_REQUEST]
    #[body = "Invalid `boundary` for `multipart/form-data` request"]
    /// Rejection type used if the `boundary` in a `multipart/form-data` is
    /// missing or invalid.
    pub struct InvalidBoundary;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        body::{self, Body},
        response::{IntoResponse, Response},
        routing::post,
        test_helpers::*,
        Router,
    };

    #[tokio::test]
    async fn content_type_with_encoding() {
        const BYTES: &[u8] = "<!doctype html><title>🦀</title>".as_bytes();
        const FILE_NAME: &str = "index.html";
        const CONTENT_TYPE: &str = "text/html; charset=utf-8";

        async fn handle(mut multipart: Multipart) -> impl IntoResponse {
            let field = multipart.next_field().await.unwrap().unwrap();

            assert_eq!(field.file_name().unwrap(), FILE_NAME);
            assert_eq!(field.content_type().unwrap(), CONTENT_TYPE);
            assert_eq!(field.bytes().await.unwrap(), BYTES);

            assert!(multipart.next_field().await.unwrap().is_none());
        }

        let app = Router::new().route("/", post(handle));

        let client = TestClient::new(app);

        let form = reqwest::multipart::Form::new().part(
            "file",
            reqwest::multipart::Part::bytes(BYTES)
                .file_name(FILE_NAME)
                .mime_str(CONTENT_TYPE)
                .unwrap(),
        );

        client.post("/").multipart(form).send().await;
    }

    // No need for this to be a #[test], we just want to make sure it compiles
    fn _multipart_from_request_limited() {
        async fn handler(_: Multipart) {}
        let _app: Router<(), http_body::Limited<Body>> = Router::new().route("/", post(handler));
    }

    #[allow(dead_code)]
    fn stream_fields_directly() {
        use futures_util::TryStreamExt;

        async fn handler(multipart: Multipart) -> Response {
            let byte_stream = multipart
                .into_stream(|field| field.into_stream())
                .try_flatten();

            body::boxed(Body::wrap_stream(byte_stream)).into_response()
        }

        let _: Router = Router::new().route("/", post(handler));
    }
}
