use actix_web::http::StatusCode;
use actix_web::{get, web, App, HttpResponse, HttpResponseBuilder, HttpServer, Responder};
use actix_web_helmet::Helmet;
use clap::Parser;
use mycal::classifier::train_qrels;
use mycal::{index, Classifier, Store};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
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
struct TrainArgs {
    model_file: String,
    qrels_file: String,
    rel_level: Option<i32>,
    sample_neg: Option<usize>,
}

#[get("/train")]
async fn train(state: web::Data<AppState>, query: web::Query<TrainArgs>) -> impl Responder {
    match train_qrels(
        &mut state.coll.lock().unwrap(),
        &query.model_file,
        &query.qrels_file,
        query.rel_level.unwrap_or(1),
        query.sample_neg.unwrap_or(0),
    ) {
        Ok(_classifier) => HttpResponse::Ok().finish(),
        Err(e) => HttpResponseBuilder::new(StatusCode::INTERNAL_SERVER_ERROR).body(e.to_string()),
    }
}

#[derive(Deserialize)]
struct ScoreArgs {
    model_file: String,
    num_results: usize,
    exclude_file: Option<String>,
}

#[get("/score")]
async fn score(state: web::Data<AppState>, query: web::Query<ScoreArgs>) -> impl Responder {
    let mut coll = state.coll.lock().unwrap();

    let exclude = match &query.exclude_file {
        Some(efn) => {
            let exclude_fp = BufReader::new(File::open(efn).expect("Can't open file"));
            exclude_fp
                .lines()
                .map(|line| line.unwrap().split_whitespace().nth(2).unwrap().to_string())
                .collect()
        }
        _ => Vec::new(),
    };

    // Convert the model into a vector of FeaturePairs.
    // The weight vector is in tokid order.
    let model = Classifier::load(&query.model_file)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err.to_string()))
        .unwrap();

    let config = index::IndexSearchConfig::new()
        .with_num_results(query.num_results)
        .with_exclude_docs(exclude);
    let rvec = index::score_using_index(&mut coll, model, config).unwrap();

    for r in rvec.iter().take(query.num_results) {
        println!("{} {}", r.docid, r.score);
    }

    HttpResponse::Ok().json(rvec.iter().take(query.num_results).collect::<Vec<_>>())
}
