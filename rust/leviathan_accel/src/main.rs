use serde::{Deserialize, Serialize};
use std::io::{self, Read};

#[derive(Debug, Serialize, Deserialize)]
struct Payload {
    input: String,
}

fn main() {
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer).ok();
    let payload: Payload = serde_json::from_str(&buffer).unwrap_or(Payload {
        input: "".to_string(),
    });

    let output = format!("Leviathan accel processed {} bytes", payload.input.len());
    let response = serde_json::json!({
        "ok": true,
        "output": output
    });
    println!("{}", response);
}
