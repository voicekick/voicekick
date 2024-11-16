use voice_stream::voice::{VoiceDetection, SILERO_VAD_CHUNK_SIZE};

use voice_tests::{
    preprocess_samples, read_voice_dataset_wav_into_samples, read_wav_into_samples,
    voice_datasets_path,
};

#[tokio::test]
async fn test_silero_vad_predict() {
    let mut vd = VoiceDetection::default().with_silero_vad_voice_threshold(0.01);

    let (samples, codec_params) = read_voice_dataset_wav_into_samples("Harvard list 01.wav");

    let sample_rate = codec_params.sample_rate.unwrap_or(0) as usize;
    let channels = codec_params
        .channels
        .map(|channels| channels.count())
        .unwrap_or(1);

    let input = preprocess_samples(sample_rate, 16000, channels, Some(512), samples);

    let chunks = input.chunks(SILERO_VAD_CHUNK_SIZE).collect::<Vec<_>>();

    // Chunk 167 result: 0.008593738
    // Chunk 168 result: 0.0069912076
    // Chunk 169 result: 0.007535875
    // Chunk 170 result: 0.007255882
    // Chunk 171 result: 0.0068237185
    // Chunk 172 result: 0.011610687
    // Chunk 173 result: 0.008559197
    // Chunk 174 result: 0.023610532
    // Chunk 175 result: 0.022555321
    // Chunk 176 result: 0.057816952
    // Chunk 177 result: 0.041098803
    // Chunk 178 result: 0.0388346
    // Chunk 179 result: 0.053876013
    // Chunk 180 result: 0.10841608
    // Chunk 181 result: 0.066301525
    // Chunk 182 result: 0.031200081
    // Chunk 183 result: 0.021389097
    // Chunk 184 result: 0.017400086
    // Chunk 185 result: 0.017270476
    // Chunk 186 result: 0.015530646
    // Chunk 187 result: 0.013654709
    // Chunk 188 result: 0.01140973
    // Chunk 189 result: 0.013352901
    // Chunk 190 result: 0.051264077
    // Chunk 191 result: 0.120171815
    assert!(!(vd.silero_vad_prediction(chunks[167].to_vec()) > 0.01));
    assert!(vd.silero_vad_prediction(chunks[190].to_vec()) > 0.01);

    let (samples, codec_params) = read_voice_dataset_wav_into_samples("obama/1.wav");

    let sample_rate = codec_params.sample_rate.unwrap_or(0) as usize;
    let channels = codec_params
        .channels
        .map(|channels| channels.count())
        .unwrap_or(1);

    let input = preprocess_samples(sample_rate, 16000, channels, Some(512), samples);

    let chunks = input.chunks(SILERO_VAD_CHUNK_SIZE);

    for chunk in chunks {
        assert!(vd.silero_vad_is_voice(chunk.to_vec()))
    }
}

#[tokio::test]
async fn test_webrtc_vad_is_noise() {
    let mut vd = VoiceDetection::default().with_silero_vad_voice_threshold(0.01);

    let (samples, codec_params) = read_voice_dataset_wav_into_samples("Harvard list 01.wav");

    let sample_rate = codec_params.sample_rate.unwrap_or(0) as usize;
    let channels = codec_params
        .channels
        .map(|channels| channels.count())
        .unwrap_or(1);

    let input = preprocess_samples(sample_rate, 16000, channels, Some(512), samples);

    let chunks = input.chunks(480).collect::<Vec<_>>();

    // WebRTC is constantly mutating it's state curiously without this loop tests will not pass
    // hence tests are not deterministic but based on the full input
    for (_i, chunk) in chunks.iter().enumerate() {
        // println!(
        //     "Chunk {} webrtc is noise {}",
        //     i,
        //     vd.webrtc_vad_is_noise(chunk)
        // );
        vd.webrtc_vad_is_noise(chunk);
    }

    // Silence
    assert!(!vd.webrtc_vad_is_noise(chunks[0]));
    // Silence
    assert!(vd.webrtc_vad_is_noise(chunks[206]));
    // Noise
    assert!(vd.webrtc_vad_is_noise(chunks[226]));
}

#[tokio::test]
async fn test_vad_combined() {
    let mut vd = VoiceDetection::default();

    let files = vec![
        voice_datasets_path("Harvard list 01.wav"),
        voice_datasets_path("obama/1.wav"),
        voice_datasets_path("24bit-M1F1-int24WE-AFsp.wav"),
    ];

    // Max 481 samples per chunk due to earshot WebRTC model limitations
    let chunk_size = 341;

    let silero_predict_treshold = 0.01; // 0.01 is a good treshold for voice detection derived from
                                        // Obama's voice dataset 1 file

    println!("Testing available VAD detection methods");
    println!("{}", "-".repeat(80));
    println!("`True voice` predict means when both WebRTC and SileroVAD predicts noise AND voice");
    println!(
        "`True noise` predict means when both WebRTC and SileroVAD predicts no noise AND no voice"
    );
    println!("{}", "-".repeat(80));

    for file in files {
        let (samples, codec_params) = read_wav_into_samples(&file);

        let sample_rate = codec_params.sample_rate.unwrap_or(0) as usize;
        let channels = codec_params
            .channels
            .map(|channels| channels.count())
            .unwrap_or(1);

        let input = preprocess_samples(sample_rate, 16000, channels, Some(512), samples);

        let chunks = input.chunks(chunk_size).collect::<Vec<_>>();

        let total_chunks = chunks.len();

        let mut true_voice: Vec<f32> = Vec::new();
        let mut true_noise: Vec<f32> = Vec::new();
        let mut times_noise = 0;
        let mut times_voice = 0;

        for (_i, chunk) in chunks.into_iter().enumerate() {
            let is_noise = vd.webrtc_vad_is_noise(chunk);
            let silero_predict = vd.silero_vad_prediction(chunk.to_vec());
            let is_voice = silero_predict > silero_predict_treshold;

            if is_noise && is_voice {
                true_voice.push(silero_predict);
            }

            if !is_noise && !is_voice {
                true_noise.push(silero_predict);
            }

            if is_noise {
                times_noise += 1;
            }

            if is_voice {
                times_voice += 1;
            }

            // println!(
            //     "Chunk {} is_noise: {} is_voice: {} silero_predict: {:.7}",
            //     i, is_noise, is_voice, silero_predict
            // );
        }

        println!("File {file} Silero threshold: {}", silero_predict_treshold);
        println!(
            "`True voice` silero predict average threshold: {}",
            true_voice.iter().sum::<f32>() / true_voice.len() as f32
        );
        println!(
            "`True noise` silero predict average threshold: {}",
            true_noise.iter().sum::<f32>() / true_noise.len() as f32
        );
        println!(
            "True Voice ratio in samples: {} Times voice ratio {} Times noise ratio {}",
            true_voice.len() as f32 / total_chunks as f32,
            times_voice as f32 / total_chunks as f32,
            times_noise as f32 / total_chunks as f32,
        );
    }
}
