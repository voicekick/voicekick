use std::time::Instant;

use dioxus::{
    hooks::{use_context, UnboundedReceiver},
    signals::{ReadableVecExt, Signal, Writable, WritableVecExt},
};
use futures_util::StreamExt;

use command_parser::{CommandOutput, CommandParser, CommandParserError};
use inference_candle::proto::Segment;
use tracing::error;

use crate::states::{VoiceCommandSegment, VoiceCommandStatus, VoiceState};

const MAX_SEGMENTS: usize = 30;

pub async fn segments_service(mut rx: UnboundedReceiver<Segment>) {
    let mut voice_state = use_context::<VoiceState>();
    let parser = use_context::<CommandParser>();

    while let Some(segment) = rx.next().await {
        let voice_text = segment.dr.text.clone();

        let mut command_text = Signal::new("N/A".into());
        let mut status = Signal::new(VoiceCommandStatus::Loading);

        let now = Instant::now();
        let mut execution_time = Signal::new(0.0);

        voice_state.segments.push(VoiceCommandSegment {
            segment,
            command_text,
            status,
            execution_time,
        });

        match parser.parse(&voice_text) {
            Ok((cmd, arg)) => {
                // Execute function
                match cmd.execute(arg).await {
                    Ok(CommandOutput::Ok(msg)) => {
                        *command_text.write() = msg.unwrap_or("Ok".into());
                        *status.write() = VoiceCommandStatus::Success;
                    }
                    Ok(CommandOutput::None) => {
                        *command_text.write() = "Ok".into();
                        *status.write() = VoiceCommandStatus::Success;
                    }
                    Err(e) => {
                        error!("Command error {:?}", e);
                        *command_text.write() = format!("Error: {}", e);
                        *status.write() = VoiceCommandStatus::Failed;
                    }
                    _ => {}
                }
            }
            Err(CommandParserError::InvalidFormat) => {
                *status.write() = VoiceCommandStatus::Failed;
            }
            Err(e) => {
                error!("Parsing error {:?}", e);
                *status.write() = VoiceCommandStatus::NoMatches;
            }
        }

        execution_time.set(now.elapsed().as_secs_f64() * 1000.0);

        // Optionally limit number of stored segments
        if voice_state.segments.len() > MAX_SEGMENTS {
            let segments = voice_state
                .segments
                .split_off(voice_state.segments.len() - MAX_SEGMENTS);
            voice_state.segments.clear();
            voice_state.segments.extend(segments);
        }
    }
}
