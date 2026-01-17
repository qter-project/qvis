use opencv::{
    Error, Result,
    core::{self, CV_8UC3, Point, Scalar, Vector},
    highgui, imgcodecs, imgproc,
    photo::{self, fast_nl_means_denoising_colored},
    prelude::*,
};
use std::sync::{Arc, Mutex};

const WINDOW_NAME: &str = "Qvis Sticker Calibration";

struct State {
    img: Mat,
    img_labeled: Mat,
    lab: Mat,
    denoised: Mat,
    edges: Mat,
    row1: Mat,
    row2: Mat,
    tmp: Mat,
    grid: Mat,
    lab_bgr: Mat,
    edges_bgr: Mat,
    all_contours_view: Mat,

    contours: Vector<Vector<Point>>,

    err: Option<Error>,
}

fn recompute(state: &mut State) -> Result<()> {
    let canny_low = highgui::get_trackbar_pos("Canny Low", WINDOW_NAME)? as f64;
    let canny_high = highgui::get_trackbar_pos("Canny High", WINDOW_NAME)? as f64;
    let h = highgui::get_trackbar_pos("H", WINDOW_NAME)? as f32;
    let h_color = highgui::get_trackbar_pos("H_color", WINDOW_NAME)? as f32;

    imgproc::cvt_color_def(&state.img, &mut state.lab, imgproc::COLOR_BGR2Lab)?;
    // photo::fast_nl_means_denoising_colored(
    //     &state.lab,
    //     &mut state.denoised,
    //     h,
    //     h_color,
    //     7,
    //     21,
    // )?;
    fast_nl_means_denoising_cielab_split(&state.lab, &mut state.denoised, h, h_color, 7, 21)?;

    imgproc::canny(
        &state.lab,
        &mut state.edges,
        canny_low,
        canny_high,
        3,
        false,
    )?;
    imgproc::find_contours_def(
        &state.edges,
        &mut state.contours,
        imgproc::RETR_LIST,
        imgproc::CHAIN_APPROX_SIMPLE,
    )?;

    state.all_contours_view =
        Mat::zeros(state.edges.rows(), state.edges.cols(), CV_8UC3)?.to_mat()?;

    state.img_labeled = state.img.clone();

    add_label(&mut state.img_labeled, "Original")?;
    add_label(&mut state.denoised, "Denoised")?;
    add_label(&mut state.lab, "Lab")?;
    add_label(&mut state.edges, "Canny Edges")?;
    add_label(
        &mut state.all_contours_view,
        &format!("All Contours ({})", state.contours.len()),
    )?;

    imgproc::cvt_color_def(&state.lab, &mut state.lab_bgr, imgproc::COLOR_Lab2BGR)?;
    imgproc::cvt_color_def(&state.edges, &mut state.edges_bgr, imgproc::COLOR_GRAY2BGR)?;

    let order = [
        &state.img_labeled,
        &state.lab_bgr,
        &state.denoised,
        &state.edges_bgr,
        &state.all_contours_view,
    ];
    core::hconcat2(order[0], order[1], &mut state.row1)?;
    core::hconcat2(&state.row1, order[2], &mut state.tmp)?;
    std::mem::swap(&mut state.row1, &mut state.tmp);
    core::hconcat2(order[3], order[4], &mut state.row2)?;
    core::hconcat2(&state.row2, order[4], &mut state.tmp)?;
    std::mem::swap(&mut state.row2, &mut state.tmp);
    core::vconcat2(&state.row1, &state.row2, &mut state.grid)?;
    highgui::imshow(WINDOW_NAME, &state.grid)?;

    Ok(())
}

fn main() -> Result<()> {
    let img = imgcodecs::imread("input.png", imgcodecs::IMREAD_COLOR)?;

    highgui::named_window(WINDOW_NAME, highgui::WINDOW_NORMAL)?;

    let state = Arc::new(Mutex::new(State {
        img: img.clone(),
        img_labeled: img.clone(),
        lab: Mat::default(),
        denoised: Mat::default(),
        edges: Mat::default(),
        row1: Mat::default(),
        row2: Mat::default(),
        tmp: Mat::default(),
        grid: Mat::default(),
        lab_bgr: Mat::default(),
        edges_bgr: Mat::default(),
        all_contours_view: Mat::default(),
        contours: Vector::new(),
        err: None,
    }));

    let make_cb = |state: Arc<Mutex<State>>| {
        Box::new(move |_pos: i32| {
            let mut state = state.lock().unwrap();
            if let Err(e) = recompute(&mut state) {
                state.err = Some(e);
            }
        })
    };

    highgui::create_trackbar(
        "Canny Low",
        WINDOW_NAME,
        None,
        300,
        Some(make_cb(state.clone())),
    )?;
    highgui::create_trackbar(
        "Canny High",
        WINDOW_NAME,
        None,
        300,
        Some(make_cb(state.clone())),
    )?;
    highgui::create_trackbar("H", WINDOW_NAME, None, 180, Some(make_cb(state.clone())))?;
    highgui::create_trackbar(
        "H_color",
        WINDOW_NAME,
        None,
        180,
        Some(make_cb(state.clone())),
    )?;

    highgui::set_trackbar_pos("Canny Low", WINDOW_NAME, 50)?;
    highgui::set_trackbar_pos("Canny High", WINDOW_NAME, 150)?;
    highgui::set_trackbar_pos("H", WINDOW_NAME, 10)?;
    highgui::set_trackbar_pos("H_color", WINDOW_NAME, 10)?;

    recompute(&mut state.lock().unwrap())?;

    loop {
        if let Some(err) = state.lock().unwrap().err.take() {
            return Err(err);
        }
        if highgui::wait_key(30)? == 27 {
            break;
        }
    }

    Ok(())
}

fn is_roughly_square(approx: &Vector<Point>, tolerance: f64) -> Result<bool> {
    let p0 = approx.get(0)?;
    let p1 = approx.get(1)?;
    let p2 = approx.get(2)?;
    let p3 = approx.get(3)?;

    let side1 = distance(p0, p1);
    let side2 = distance(p1, p2);
    let side3 = distance(p2, p3);
    let side4 = distance(p3, p0);

    let avg_side = (side1 + side2 + side3 + side4) / 4.0;

    let tolerance_amount = avg_side * tolerance;
    let is_square = (side1 - avg_side).abs() < tolerance_amount
        && (side2 - avg_side).abs() < tolerance_amount
        && (side3 - avg_side).abs() < tolerance_amount
        && (side4 - avg_side).abs() < tolerance_amount;

    Ok(is_square)
}

fn distance(p1: Point, p2: Point) -> f64 {
    let dx = (p1.x - p2.x) as f64;
    let dy = (p1.y - p2.y) as f64;
    (dx * dx + dy * dy).sqrt()
}

fn add_label(img: &mut Mat, text: &str) -> Result<()> {
    imgproc::put_text(
        img,
        text,
        Point::new(10, 80),
        imgproc::FONT_HERSHEY_SIMPLEX,
        3.0,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        4,
        imgproc::LINE_8,
        false,
    )?;
    Ok(())
}

// let mut lab = Mat::default();
// imgproc::cvt_color_def(&img, &mut lab, imgproc::COLOR_BGR2Lab)?;
pub fn fast_nl_means_denoising_cielab_split(
    lab: &Mat,
    dst: &mut Mat,
    h: f32,
    h_color: f32,
    template_window_size: i32,
    search_window_size: i32,
) -> Result<()> {
    let mut channels = Vector::<Mat>::new();
    core::split(lab, &mut channels)?;

    let a = channels.get(1)?.clone();
    let b = channels.get(2)?.clone();

    // let mut l_dn = Mat::default();
    let mut a_dn = Mat::default();
    let mut b_dn = Mat::default();

    // photo::fast_nl_means_denoising(
    //     &l,
    //     &mut l_dn,
    //     h,
    //     template_window_size,
    //     search_window_size,
    // )?;

    photo::fast_nl_means_denoising(
        &a,
        &mut a_dn,
        h_color,
        template_window_size,
        search_window_size,
    )?;

    photo::fast_nl_means_denoising(
        &b,
        &mut b_dn,
        h_color,
        template_window_size,
        search_window_size,
    )?;

    // let mut merged = Vector::<Mat>::new();
    // merged.push(l_dn);
    // merged.push(a_dn);
    // merged.push(b_dn);
    channels.set(1, a_dn)?;
    channels.set(2, b_dn)?;

    core::merge(&channels, dst)?;

    Ok(())
}
