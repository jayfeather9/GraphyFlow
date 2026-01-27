use std::cmp::max;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use std::io::Write;

use clap::{ArgGroup, Parser, ValueEnum};
use memmap2::Mmap;
use serde::Deserialize;

const INF_U32: u32 = u32::MAX / 4;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum IndexingMode {
    Auto,
    ZeroBased,
    OneBased,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum WeightMode {
    Unit,
    ThirdColumnU32,
}

#[derive(Parser, Debug)]
#[command(
    name = "graphyflow-sssp",
    about = "Fast one-pass SSSP/new_dist-style relaxation on an edge list"
)]
#[command(group(
    ArgGroup::new("input")
        .required(true)
        .args(["plan", "dataset"]),
))]
struct Args {
    /// JSON plan file. When provided, CLI flags become defaults overridden by the plan.
    #[arg(long)]
    plan: Option<PathBuf>,

    /// Path to edge list dataset (2 or 3 integers per line).
    #[arg(long)]
    dataset: Option<PathBuf>,

    /// Source node id (0-based after indexing adjustment).
    #[arg(long, default_value_t = 0)]
    source: u32,

    /// Node count (optional). If omitted, grows dynamically based on max id observed.
    #[arg(long)]
    num_nodes: Option<usize>,

    /// Input indexing mode.
    #[arg(long, value_enum, default_value_t = IndexingMode::Auto)]
    indexing: IndexingMode,

    /// Weight interpretation.
    #[arg(long, value_enum, default_value_t = WeightMode::Unit)]
    weight: WeightMode,

    /// Sample bytes used to auto-detect indexing when `--indexing auto`.
    #[arg(long, default_value_t = 32 * 1024 * 1024)]
    detect_sample_bytes: usize,

    /// Print JSON summary (default).
    #[arg(long, default_value_t = true)]
    json: bool,

    /// Optional output file for results (binary).
    /// - `--out-format full-u32-le`: writes `nodes` u32 distances (little-endian)
    /// - `--out-format updates-u32x2-le`: writes pairs `(node_id, dist)` for all dst nodes seen in the edge list
    #[arg(long)]
    out: Option<PathBuf>,

    /// Output format when `--out` is provided.
    #[arg(long, value_enum, default_value_t = OutFormat::FullU32Le)]
    out_format: OutFormat,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum OutFormat {
    FullU32Le,
    UpdatesU32x2Le,
}

#[derive(Debug, Deserialize)]
struct Plan {
    dataset: PathBuf,
    #[serde(default)]
    source: Option<u32>,
    #[serde(default)]
    num_nodes: Option<usize>,
    #[serde(default)]
    indexing: Option<IndexingModeSerde>,
    #[serde(default)]
    weight: Option<WeightModeSerde>,
    #[serde(default)]
    detect_sample_bytes: Option<usize>,
    #[serde(default)]
    out: Option<PathBuf>,
    #[serde(default)]
    out_format: Option<OutFormatSerde>,
}

#[derive(Copy, Clone, Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum IndexingModeSerde {
    Auto,
    ZeroBased,
    OneBased,
}

#[derive(Copy, Clone, Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WeightModeSerde {
    Unit,
    ThirdColumnU32,
}

#[derive(Copy, Clone, Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum OutFormatSerde {
    FullU32Le,
    UpdatesU32x2Le,
}

impl From<IndexingModeSerde> for IndexingMode {
    fn from(v: IndexingModeSerde) -> Self {
        match v {
            IndexingModeSerde::Auto => IndexingMode::Auto,
            IndexingModeSerde::ZeroBased => IndexingMode::ZeroBased,
            IndexingModeSerde::OneBased => IndexingMode::OneBased,
        }
    }
}

impl From<WeightModeSerde> for WeightMode {
    fn from(v: WeightModeSerde) -> Self {
        match v {
            WeightModeSerde::Unit => WeightMode::Unit,
            WeightModeSerde::ThirdColumnU32 => WeightMode::ThirdColumnU32,
        }
    }
}

impl From<OutFormatSerde> for OutFormat {
    fn from(v: OutFormatSerde) -> Self {
        match v {
            OutFormatSerde::FullU32Le => OutFormat::FullU32Le,
            OutFormatSerde::UpdatesU32x2Le => OutFormat::UpdatesU32x2Le,
        }
    }
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\n' | b'\r' | b'\t')
}

fn skip_ws(buf: &[u8], mut i: usize) -> usize {
    while i < buf.len() && is_ws(buf[i]) {
        i += 1;
    }
    i
}

fn read_u32(buf: &[u8], mut i: usize) -> Option<(u32, usize)> {
    i = skip_ws(buf, i);
    if i >= buf.len() {
        return None;
    }
    let mut v: u32 = 0;
    let mut saw_digit = false;
    while i < buf.len() {
        let b = buf[i];
        if b.is_ascii_digit() {
            saw_digit = true;
            v = v
                .saturating_mul(10)
                .saturating_add((b - b'0') as u32);
            i += 1;
        } else {
            break;
        }
    }
    if !saw_digit {
        return None;
    }
    Some((v, i))
}

fn detect_indexing(buf: &[u8], sample_bytes: usize) -> (i32, Duration) {
    let start = Instant::now();
    let limit = buf.len().min(sample_bytes);
    let mut i = 0usize;
    let mut saw_zero = false;
    let mut min_id: u32 = u32::MAX;

    while i < limit {
        let (u, i1) = match read_u32(buf, i) {
            Some(x) => x,
            None => break,
        };
        let (v, i2) = match read_u32(buf, i1) {
            Some(x) => x,
            None => break,
        };
        i = i2;

        if u == 0 || v == 0 {
            saw_zero = true;
            break;
        }
        min_id = min_id.min(u).min(v);

        // Optionally skip a third column if present on this line (best-effort).
        // This keeps detection robust for weighted edge lists.
        let j = skip_ws(buf, i);
        if j < limit && buf[j].is_ascii_digit() {
            if let Some((_w, j2)) = read_u32(buf, j) {
                i = j2;
            }
        }
        // Move to next line quickly
        while i < limit && buf[i] != b'\n' {
            i += 1;
        }
        i = skip_ws(buf, i);
    }

    let offset = if saw_zero {
        0
    } else if min_id == 1 {
        1
    } else {
        0
    };

    (offset, start.elapsed())
}

fn ensure_len(dist: &mut Vec<u32>, needed: usize) {
    if needed < dist.len() {
        return;
    }
    let mut new_len = max(dist.len().saturating_mul(2), 1024);
    if new_len <= needed {
        new_len = needed + 1;
    }
    dist.resize(new_len, INF_U32);
}

fn ensure_len_u8(buf: &mut Vec<u8>, needed: usize) {
    if needed < buf.len() {
        return;
    }
    let mut new_len = max(buf.len().saturating_mul(2), 1024);
    if new_len <= needed {
        new_len = needed + 1;
    }
    buf.resize(new_len, 0);
}

fn load_plan(path: &PathBuf) -> anyhow::Result<Plan> {
    let bytes = std::fs::read(path)?;
    let plan: Plan = serde_json::from_slice(&bytes)?;
    Ok(plan)
}

fn main() -> anyhow::Result<()> {
    // Keep main small; errors should be clear.
    let mut args = Args::parse();

    if let Some(plan_path) = &args.plan {
        let plan = load_plan(plan_path)?;
        args.dataset = Some(plan.dataset);
        if let Some(v) = plan.source {
            args.source = v;
        }
        if let Some(v) = plan.num_nodes {
            args.num_nodes = Some(v);
        }
        if let Some(v) = plan.indexing {
            args.indexing = v.into();
        }
        if let Some(v) = plan.weight {
            args.weight = v.into();
        }
        if let Some(v) = plan.detect_sample_bytes {
            args.detect_sample_bytes = v;
        }
        if let Some(v) = plan.out {
            args.out = Some(v);
        }
        if let Some(v) = plan.out_format {
            args.out_format = v.into();
        }
    }

    let dataset = args
        .dataset
        .clone()
        .ok_or_else(|| anyhow::anyhow!("Missing dataset (provide --dataset or --plan with dataset)"))?;

    let file = std::fs::File::open(&dataset)?;
    let mmap = unsafe { Mmap::map(&file)? };

    let (offset, detect_dur) = match args.indexing {
        IndexingMode::Auto => detect_indexing(&mmap, args.detect_sample_bytes),
        IndexingMode::ZeroBased => (0, Duration::from_secs(0)),
        IndexingMode::OneBased => (1, Duration::from_secs(0)),
    };

    let start = Instant::now();

    let mut dist: Vec<u32> = Vec::new();
    if let Some(n) = args.num_nodes {
        dist.resize(n, INF_U32);
    }
    ensure_len(&mut dist, args.source as usize);
    dist[args.source as usize] = 0;

    // One-pass relaxation: new_dist starts as dist (but our default init has only source finite).
    let mut new_dist = dist.clone();
    let mut seen_dst: Vec<u8> = Vec::new();

    let mut i = 0usize;
    let mut edges: u64 = 0;
    let mut updates: u64 = 0;
    let mut max_node: u32 = (dist.len().saturating_sub(1)) as u32;

    while let Some((mut u, i1)) = read_u32(&mmap, i) {
        let (mut v, i2) = match read_u32(&mmap, i1) {
            Some(x) => x,
            None => break,
        };
        i = i2;

        if offset == 1 {
            if u == 0 || v == 0 {
                return Err(anyhow::anyhow!(
                    "Encountered 0 id after applying 1-based indexing adjustment"
                ));
            }
            u -= 1;
            v -= 1;
        }

        let w: u32 = match args.weight {
            WeightMode::Unit => 1,
            WeightMode::ThirdColumnU32 => {
                // Best-effort: if missing, default to 1
                if let Some((ww, i3)) = read_u32(&mmap, i) {
                    i = i3;
                    ww
                } else {
                    1
                }
            }
        };

        let u_usize = u as usize;
        let v_usize = v as usize;
        ensure_len(&mut dist, u_usize);
        ensure_len(&mut dist, v_usize);
        ensure_len(&mut new_dist, v_usize);
        ensure_len_u8(&mut seen_dst, v_usize);
        seen_dst[v_usize] = 1;

        if u > max_node {
            max_node = u;
        }
        if v > max_node {
            max_node = v;
        }

        let du = dist[u_usize];
        if du != INF_U32 {
            let cand = du.saturating_add(w);
            if cand < new_dist[v_usize] {
                new_dist[v_usize] = cand;
                updates += 1;
            }
        }

        edges += 1;

        // Skip to end of line quickly to avoid scanning trailing junk.
        while i < mmap.len() && mmap[i] != b'\n' {
            i += 1;
        }
        i = skip_ws(&mmap, i);
    }

    let dur = start.elapsed();

    // Tighten node count to max observed (+1), unless user provided one.
    let nodes = if let Some(n) = args.num_nodes {
        n
    } else {
        (max_node as usize) + 1
    };

    if let Some(out_path) = &args.out {
        let mut f = std::io::BufWriter::new(std::fs::File::create(out_path)?);
        match args.out_format {
            OutFormat::FullU32Le => {
                let end = nodes.min(new_dist.len());
                for &d in &new_dist[..end] {
                    f.write_all(&d.to_le_bytes())?;
                }
            }
            OutFormat::UpdatesU32x2Le => {
                let end = nodes.min(new_dist.len()).min(seen_dst.len());
                for node_id in 0..end {
                    if seen_dst[node_id] == 0 {
                        continue;
                    }
                    let nid = node_id as u32;
                    let d = new_dist[node_id];
                    f.write_all(&nid.to_le_bytes())?;
                    f.write_all(&d.to_le_bytes())?;
                }
            }
        }
        f.flush()?;
    }

    if args.json {
        // Manual JSON to avoid pulling serde for now.
        println!(
            "{{\"dataset\":\"{}\",\"indexing_applied\":{},\"edges\":{},\"nodes\":{},\"updates\":{},\"detect_sec\":{:.6},\"parse_and_compute_sec\":{:.6}}}",
            dataset.display(),
            offset,
            edges,
            nodes,
            updates,
            detect_dur.as_secs_f64(),
            dur.as_secs_f64(),
        );
    } else {
        println!("dataset: {}", dataset.display());
        println!("indexing_applied: {}", offset);
        println!("edges: {}", edges);
        println!("nodes: {}", nodes);
        println!("updates: {}", updates);
        println!("detect_sec: {:.6}", detect_dur.as_secs_f64());
        println!("parse_and_compute_sec: {:.6}", dur.as_secs_f64());
    }

    Ok(())
}
