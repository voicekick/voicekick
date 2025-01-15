use dioxus::hooks::UnboundedReceiver;
use futures_util::StreamExt;
use inference_candle::proto::Segment;

pub async fn segments_service(mut rx: UnboundedReceiver<Segment>) {
    while let Some(segment) = rx.next().await {
        println!("Received segment: {:?}", segment);
    }
}
