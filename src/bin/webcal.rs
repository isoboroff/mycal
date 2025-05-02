use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use actix_web_helmet::Helmet;
use clap::Parser;
use mycal::Store;
use std::sync::Mutex;

#[derive(Parser)]
struct Cli {
    coll_prefix: String,
}

struct AppState {
    coll: Mutex<Store>,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args = Cli::parse();
    let coll = web::Data::new(AppState {
        coll: Store::open(&args.coll_prefix)?.into(),
    });

    HttpServer::new(move || {
        App::new()
            .wrap(Helmet::default())
            .app_data(coll.clone())
            .service(train)
            .service(score)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}

#[get("/train")]
async fn train(state: web::Data<AppState>) -> impl Responder {
    HttpResponse::Ok().body(format!("train sez {:?}", state.coll.lock().unwrap().config))
}

#[post("/score")]
async fn score(state: web::Data<AppState>) -> impl Responder {
    HttpResponse::Ok().body(format!("score sez {:?}", state.coll.lock().unwrap().config))
}
