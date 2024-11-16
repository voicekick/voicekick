// Custom trait to convert any sample type to f32
pub trait IntoF32 {
    fn into_f32(self) -> f32;
}

impl IntoF32 for i8 {
    fn into_f32(self) -> f32 {
        self as f32 / i8::MAX as f32
    }
}

impl IntoF32 for i16 {
    fn into_f32(self) -> f32 {
        self as f32 / i16::MAX as f32
    }
}

impl IntoF32 for i32 {
    fn into_f32(self) -> f32 {
        self as f32 / i32::MAX as f32
    }
}

impl IntoF32 for f32 {
    fn into_f32(self) -> f32 {
        self // Already in f32 format
    }
}

pub trait IntoI16 {
    fn into_i16(self) -> i16;
}

impl IntoI16 for f32 {
    fn into_i16(self) -> i16 {
        (self * i16::MAX as f32) as i16
    }
}
