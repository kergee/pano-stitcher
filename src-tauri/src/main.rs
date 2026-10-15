#[cfg(not(debug_assertions))]
use std::sync::Mutex;
#[cfg(not(debug_assertions))]
use std::time::Duration;

#[cfg(not(debug_assertions))]
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
#[cfg(not(debug_assertions))]
use tauri_plugin_shell::ShellExt;
#[cfg(not(debug_assertions))]
use tauri::Manager;

#[cfg(not(debug_assertions))]
#[allow(dead_code)]
struct BackendProcess(Mutex<Option<CommandChild>>);

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            #[cfg(debug_assertions)]
            let _ = app;

            #[cfg(not(debug_assertions))]
            {
                let sidecar = app
                    .shell()
                    .sidecar("pano-backend")?
                    .args(["--host", "127.0.0.1", "--port", "8000"]);
                let (mut rx, child) = sidecar.spawn()?;
                app.manage(BackendProcess(Mutex::new(Some(child))));

                tauri::async_runtime::spawn(async move {
                    while let Some(event) = rx.recv().await {
                        match event {
                            CommandEvent::Stdout(line) => {
                                println!("[pano-backend] {}", String::from_utf8_lossy(&line));
                            }
                            CommandEvent::Stderr(line) => {
                                eprintln!("[pano-backend] {}", String::from_utf8_lossy(&line));
                            }
                            _ => {}
                        }
                    }
                });

                std::thread::sleep(Duration::from_millis(900));
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Pano");
}
