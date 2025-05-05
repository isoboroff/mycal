use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use actix_web_helmet::Helmet;
use clap::Parser;
use mycal::classifier::train_qrels;
use mycal::Store;
use serde::Deserialize;
use std::sync::Mutex;

#[derive(Parser)]
struct Cli {
    coll_prefix: String,
    #[arg(short, long)]
    host: Option<String>,
    #[arg(short, long)]
    port: Option<u16>,
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
    .bind((
        args.host.unwrap_or("127.0.0.1".to_string()),
        args.port.unwrap_or(8080),
    ))?
    .run()
    .await
}

#[derive(Deserialize)]
struct TrainInfo {
    coll: String,
    model_file: String,
    qrels_file: String,
    rel_level: Option<i32>,
    sample_neg: Option<usize>,
}

#[get("/train")]
async fn train(state: web::Data<AppState>, query: web::Query<TrainInfo>) -> impl Responder {
    _ = train_qrels(
        &query.coll,
        &query.model_file,
        &query.qrels_file,
        query.rel_level.unwrap_or(1),
        query.sample_neg.unwrap_or(0),
    );
    HttpResponse::Ok().body(format!("train sez {:?}", state.coll.lock().unwrap().config))
}

#[post("/score")]
async fn score(state: web::Data<AppState>) -> impl Responder {
    HttpResponse::Ok().body(format!("score sez {:?}", state.coll.lock().unwrap().config))
}
