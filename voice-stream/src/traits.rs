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

impl IntoF32 for i64 {
    fn into_f32(self) -> f32 {
        self as f32 / i64::MAX as f32
    }
}

impl IntoF32 for u8 {
    fn into_f32(self) -> f32 {
        self as f32 / u8::MAX as f32
    }
}

impl IntoF32 for u16 {
    fn into_f32(self) -> f32 {
        self as f32 / u16::MAX as f32
    }
}

impl IntoF32 for u32 {
    fn into_f32(self) -> f32 {
        self as f32 / u32::MAX as f32
    }
}

impl IntoF32 for u64 {
    fn into_f32(self) -> f32 {
        self as f32 / u64::MAX as f32
    }
}

impl IntoF32 for f32 {
    fn into_f32(self) -> f32 {
        self // Already in f32 format
    }
}

impl IntoF32 for f64 {
    fn into_f32(self) -> f32 {
        self as f32
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
