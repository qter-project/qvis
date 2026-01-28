use internment::ArcIntern;
use opencv::{
    core::{BORDER_CONSTANT, CV_8UC1, CV_8UC3, Point, Rect, Scalar, Size, Vec3b},
    highgui, imgcodecs,
    imgproc::{self, FILLED, FLOODFILL_FIXED_RANGE, FLOODFILL_MASK_ONLY, LINE_8, MORPH_ELLIPSE},
    prelude::*,
};
use puzzle_theory::puzzle_geometry::{Face, PuzzleGeometry};
use qvis::Pixel;
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};
use std::{
    f64::consts::PI,
    sync::{Arc, Mutex},
};

const WINDOW_NAME: &str = "Qvis Sticker Assignment";
const EROSION_SIZE_TRACKBAR_NAME: &str = "Erosion size";
const EROSION_SIZE_TRACKBAR_MINDEFMAX: [i32; 3] = [2, 4, 20];
const UPPER_DIFF_TRACKBAR_NAME: &str = "Upper diff";
const UPPER_DIFF_TRACKBAR_MINDEFMAX: [i32; 3] = [0, 2, 5];
const SUBMIT_BUTTON_NAME: &str = "Assign sticker";
const EROSION_KERNEL_MORPH_SHAPE: i32 = MORPH_ELLIPSE;
const DEF_ANCHOR: Point = Point::new(-1, -1);
const XY_CIRCLE_RADIUS: i32 = 6;
const MAX_PIXEL_VALUE: i32 = 255;
const MAX_PIXEL_COUNT: i32 = 500_000 * 100;
const ERODE_UNTIL_PERCENT: (i32, i32) = (1, 3);
const MIN_SAMPLES: i32 = 30;
const NUM_QVIS_PIXELS: usize = 20;

enum UIState {
    OpenCVError(opencv::Error),
    Assigning,
    Finished,
}

struct State {
    img: Mat,
    tmp_mask: Mat,
    grayscale_mask: Mat,
    cleaned_grayscale_mask: Mat,
    eroded_grayscale_mask: Mat,
    erosion_kernel: Mat,
    erosion_kernel_times_two: Mat,
    displayed_img: Mat,
    mask_roi: Rect,
    pixel_assignment: Box<[Pixel]>,
    work: Vec<(Face, Vec<ArcIntern<str>>)>,
    current_sticker_idx: usize,
    upper_flood_fill_diff: i32,
    maybe_drag_origin: Option<(i32, i32)>,
    maybe_drag_xy: Option<(i32, i32)>,
    maybe_xy: Option<(i32, i32)>,
    dragging: bool,
    ui: UIState,
}

fn c(x: i32, n: i32) -> i32 {
    (x + n) / 6
}

fn perm6_from_number(mut n: u16) -> [i32; 6] {
    const FACT: [u16; 7] = [1, 1, 2, 6, 24, 120, 720];
    n %= FACT[6];

    let mut elems = vec![0, 1, 2, 3, 4, 5];
    let mut result = [0; 6];

    for i in 0..6 {
        let f = FACT[5 - i];
        let idx = (n / f) as usize;
        n %= f;

        result[i] = elems.remove(idx);
    }

    result
}

fn update_display(state: &mut State) -> opencv::Result<()> {
    state.img.copy_to(&mut state.displayed_img)?;
    let ran;
    let shuffled;
    let mut nonzeroes: Vec<usize>;
    if let Some((drag_origin_x, drag_origin_y)) = state.maybe_drag_origin
        && let Some((drag_x, drag_y)) = state.maybe_drag_xy
    {
        ran = true;
        #[allow(clippy::cast_possible_truncation)]
        let distance = (f64::from(drag_x - drag_origin_x)
            .hypot(f64::from(drag_y - drag_origin_y))
            .powf(1.5)
            / 20.0) as i32;
        // angle is between [-pi, pi]; add pi and multiply by 360/pi to get a range
        // of [0, 720] throughout the full circle which is 6!
        //
        // multiply it again by 20 to increase the periodicity
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let angle = (f64::from(drag_y - drag_origin_y).atan2(f64::from(drag_x - drag_origin_x))
            + PI * 360.0 / PI * 20.0) as u16;
        let perm6 = perm6_from_number(angle);

        Mat::roi_mut(&mut state.grayscale_mask, state.mask_roi)?.set_to_def(&Scalar::all(0.0))?;
        imgproc::flood_fill_mask(
            &mut state.img,
            &mut state.grayscale_mask,
            Point::new(drag_origin_x, drag_origin_y),
            Scalar::default(), // ignored
            &mut Rect::default(),
            Scalar::from((
                c(distance, perm6[0]),
                c(distance, perm6[1]),
                c(distance, perm6[2]),
            )),
            Scalar::from((
                c(
                    distance,
                    perm6[3]
                        + state.upper_flood_fill_diff * MAX_PIXEL_VALUE
                            / UPPER_DIFF_TRACKBAR_MINDEFMAX[2],
                ),
                c(
                    distance,
                    perm6[4]
                        + state.upper_flood_fill_diff * MAX_PIXEL_VALUE
                            / UPPER_DIFF_TRACKBAR_MINDEFMAX[2],
                ),
                c(
                    distance,
                    perm6[5]
                        + state.upper_flood_fill_diff * MAX_PIXEL_VALUE
                            / UPPER_DIFF_TRACKBAR_MINDEFMAX[2],
                ),
            )),
            4 | FLOODFILL_FIXED_RANGE | FLOODFILL_MASK_ONLY | (MAX_PIXEL_VALUE << 8),
        )?;

        imgproc::erode(
            &state.grayscale_mask,
            &mut state.cleaned_grayscale_mask,
            &state.erosion_kernel,
            DEF_ANCHOR,
            2,
            BORDER_CONSTANT,
            imgproc::morphology_default_border_value()?,
        )?;
        let to_dilate = if opencv::core::has_non_zero(&Mat::roi(
            &state.cleaned_grayscale_mask,
            state.mask_roi,
        )?)? {
            *state
                .cleaned_grayscale_mask
                .at_2d_mut::<u8>(drag_origin_y + 1, drag_origin_x + 1)? =
                MAX_PIXEL_VALUE.try_into().unwrap();

            Mat::roi_mut(&mut state.tmp_mask, state.mask_roi)?.set_to_def(&Scalar::all(0.0))?;
            let mut cleaned_grayscale_mask_cropped_mut =
                Mat::roi_mut(&mut state.cleaned_grayscale_mask, state.mask_roi)?;
            imgproc::flood_fill_mask(
                &mut cleaned_grayscale_mask_cropped_mut,
                &mut state.tmp_mask,
                Point::new(drag_origin_x, drag_origin_y),
                Scalar::default(), // ignored
                &mut Rect::default(),
                Scalar::all(0.0),
                Scalar::all(0.0),
                4 | FLOODFILL_FIXED_RANGE | FLOODFILL_MASK_ONLY | (MAX_PIXEL_VALUE << 8),
            )?;
            std::mem::swap(&mut state.cleaned_grayscale_mask, &mut state.tmp_mask);
            &mut state.cleaned_grayscale_mask
        } else {
            &mut state.grayscale_mask
        };

        let rows = to_dilate.rows();
        let cols = to_dilate.cols();
        to_dilate
            .roi_mut(Rect::new(0, 0, cols, 2))?
            .set_to_def(&Scalar::all(0.0))?;
        to_dilate
            .roi_mut(Rect::new(0, rows - 2, cols, 2))?
            .set_to_def(&Scalar::all(0.0))?;
        to_dilate
            .roi_mut(Rect::new(0, 2, 2, rows - 4))?
            .set_to_def(&Scalar::all(0.0))?;
        to_dilate
            .roi_mut(Rect::new(cols - 2, 2, 2, rows - 4))?
            .set_to_def(&Scalar::all(0.0))?;
        // For some reason dilation doesn't work on ROIs
        imgproc::dilate(
            to_dilate,
            &mut state.tmp_mask,
            &state.erosion_kernel_times_two,
            DEF_ANCHOR,
            1,
            BORDER_CONSTANT,
            imgproc::morphology_default_border_value()?,
        )?;
        std::mem::swap(&mut state.cleaned_grayscale_mask, &mut state.tmp_mask);

        let og_num_pixels = opencv::core::count_non_zero(&state.cleaned_grayscale_mask)?;
        let mut erosion_count = 0;
        let mask_to_randomly_sample = loop {
            let has_eroded_enough = |to_check| -> Result<bool, opencv::Error> {
                let current_num_pixels = opencv::core::count_non_zero(to_check)?;
                Ok(current_num_pixels
                    <= og_num_pixels * ERODE_UNTIL_PERCENT.0 / ERODE_UNTIL_PERCENT.1
                    || current_num_pixels <= MIN_SAMPLES)
            };
            let to_erode = if erosion_count == 0 {
                if has_eroded_enough(&state.cleaned_grayscale_mask)? {
                    state
                        .cleaned_grayscale_mask
                        .copy_to(&mut state.eroded_grayscale_mask)?;
                    break &state.cleaned_grayscale_mask;
                }
                &state.cleaned_grayscale_mask
            } else {
                if has_eroded_enough(&state.eroded_grayscale_mask)? {
                    std::mem::swap(&mut state.eroded_grayscale_mask, &mut state.tmp_mask);
                    break &state.eroded_grayscale_mask;
                }
                &state.eroded_grayscale_mask
            };

            imgproc::erode(
                to_erode,
                &mut state.tmp_mask,
                &state.erosion_kernel,
                DEF_ANCHOR,
                2,
                BORDER_CONSTANT,
                imgproc::morphology_default_border_value()?,
            )?;

            if erosion_count == 0 {
                state.tmp_mask.copy_to(&mut state.eroded_grayscale_mask)?;
            } else {
                std::mem::swap(&mut state.eroded_grayscale_mask, &mut state.tmp_mask);
            }
            erosion_count += 1;
        };

        let mask_to_randomly_sample =
            Mat::roi(mask_to_randomly_sample, state.mask_roi)?.clone_pointee();

        let mut seed = [0; 32];
        seed[0..4].copy_from_slice(&drag_origin_x.to_be_bytes());
        seed[4..8].copy_from_slice(&drag_origin_y.to_be_bytes());
        let mut rng = SmallRng::from_seed(seed);
        nonzeroes = mask_to_randomly_sample
            .data_bytes()?
            .iter()
            .enumerate()
            .filter_map(|(i, &value)| {
                if value == u8::try_from(MAX_PIXEL_VALUE).unwrap() {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        shuffled = nonzeroes.partial_shuffle(&mut rng, NUM_QVIS_PIXELS).0;

        imgproc::line(
            &mut state.displayed_img,
            Point::new(drag_origin_x, drag_origin_y),
            Point::new(drag_x, drag_y),
            Scalar::all(f64::from(MAX_PIXEL_VALUE)),
            3,
            LINE_8,
            0,
        )?;
        imgproc::circle(
            &mut state.displayed_img,
            Point::new(drag_x, drag_y),
            XY_CIRCLE_RADIUS,
            Scalar::all(f64::from(MAX_PIXEL_VALUE)),
            FILLED,
            LINE_8,
            0,
        )?;
        imgproc::circle(
            &mut state.displayed_img,
            Point::new(drag_x, drag_y),
            XY_CIRCLE_RADIUS - 3,
            Scalar::from((0, 0, MAX_PIXEL_VALUE)),
            FILLED,
            LINE_8,
            0,
        )?;
    } else {
        ran = false;
        shuffled = &mut [];
    }
    imgproc::put_text(
        &mut state.displayed_img,
        &format!(
            "Choose {} on {}",
            state.work[state.current_sticker_idx]
                .1
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<String>(),
            state.work[state.current_sticker_idx].0.color
        ),
        Point::new(10, 40),
        imgproc::FONT_HERSHEY_SIMPLEX,
        1.1,
        Scalar::all(0.0),
        5,
        imgproc::LINE_8,
        false,
    )?;
    imgproc::put_text(
        &mut state.displayed_img,
        &format!(
            "Choose {} on {}",
            state.work[state.current_sticker_idx]
                .1
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<String>(),
            state.work[state.current_sticker_idx].0.color
        ),
        Point::new(10, 40),
        imgproc::FONT_HERSHEY_SIMPLEX,
        1.1,
        Scalar::all(f64::from(MAX_PIXEL_VALUE)),
        2,
        imgproc::LINE_8,
        false,
    )?;
    if ran {
        let cleaned_grayscale_mask_cropped =
            Mat::roi(&state.cleaned_grayscale_mask, state.mask_roi)?;
        state.displayed_img.set_to(
            &Scalar::from((MAX_PIXEL_VALUE, 0, MAX_PIXEL_VALUE)),
            &cleaned_grayscale_mask_cropped,
        )?;

        let eroded_grayscale_mask_cropped = Mat::roi(&state.eroded_grayscale_mask, state.mask_roi)?;
        state.displayed_img.set_to(
            &Scalar::from((MAX_PIXEL_VALUE * 3 / 4, 0, MAX_PIXEL_VALUE * 3 / 4)),
            &eroded_grayscale_mask_cropped,
        )?;

        let displayed_image_data_bytes_mut: &mut [Vec3b] = state.displayed_img.data_typed_mut()?;
        // let cols = state.img.cols() as usize;
        // dbg!(displayed_image_data_bytes_mut.len());
        // dbg!(cols);
        // dbg!(state.img.rows() as usize + 1);
        // dbg!(&state.samples);
        for i in shuffled.iter().copied() {
            // dbg!(i, cols);
            // let row = i / (cols + 0);
            // let num_padding_pixels = 2 + 4 * (row - 3);
            let num_padding_pixels = 0;
            displayed_image_data_bytes_mut[i - num_padding_pixels] = Vec3b::from_array([
                u8::try_from(MAX_PIXEL_VALUE).unwrap() / 2,
                0,
                u8::try_from(MAX_PIXEL_VALUE).unwrap() / 2,
            ]);
        }
    }
    highgui::imshow(WINDOW_NAME, &state.displayed_img)?;
    Ok(())
}

fn mouse_callback(state: &mut State, event: i32, x: i32, y: i32) -> opencv::Result<()> {
    if event == highgui::EVENT_MOUSEMOVE {
        state.maybe_xy = Some((x, y));
        if state.dragging {
            state.maybe_drag_xy = Some((x, y));
            update_display(state)?;
        }
    }

    Ok(())
}

fn erosion_kernel_trackbar_callback(state: &mut State, pos: i32) -> opencv::Result<()> {
    state.erosion_kernel =
        imgproc::get_structuring_element_def(EROSION_KERNEL_MORPH_SHAPE, Size::new(pos, pos))?;
    state.erosion_kernel_times_two = imgproc::get_structuring_element_def(
        EROSION_KERNEL_MORPH_SHAPE,
        Size::new(pos * 2, pos * 2),
    )?;
    update_display(state)?;
    Ok(())
}

fn light_tolerance_trackbar_callback(state: &mut State, pos: i32) -> opencv::Result<()> {
    state.upper_flood_fill_diff = pos;
    update_display(state)?;
    Ok(())
}

fn submit_button_callback(state: &mut State) -> opencv::Result<()> {
    let cleaned_grayscale_mask_cropped = Mat::roi(&state.cleaned_grayscale_mask, state.mask_roi)?;
    assert_eq!(
        cleaned_grayscale_mask_cropped.total(),
        state.pixel_assignment.len()
    );

    let h = cleaned_grayscale_mask_cropped.rows();
    let w = cleaned_grayscale_mask_cropped.cols();
    let mut count = 0;
    for y in 0..h {
        for x in 0..w {
            let value = *cleaned_grayscale_mask_cropped.at_2d::<u8>(y, x)?;
            let idx = usize::try_from(y * w + x).unwrap();
            if i32::from(value) == MAX_PIXEL_VALUE {
                count += 1;
                state.pixel_assignment[idx] = Pixel::Sticker(state.current_sticker_idx);
            }
        }
    }

    leptos::logging::log!(
        "Assigned {} pixels to sticker {}",
        count,
        state.current_sticker_idx
    );

    state.current_sticker_idx += 1;
    if state.current_sticker_idx == state.work.len() {
        state.ui = UIState::Finished;
    }
    state.maybe_drag_origin = None;
    update_display(state)?;

    Ok(())
}

fn restart_button_callback(state: &mut State) -> opencv::Result<()> {
    state.current_sticker_idx = 0;
    state.pixel_assignment.fill(Pixel::Unassigned);
    state.maybe_drag_origin = None;
    update_display(state)?;
    Ok(())
}

fn toggle_dragging(state: &mut State) {
    if state.dragging {
        state.dragging = false;
    } else if let Some((x, y)) = state.maybe_xy {
        if let Some((drag_x, drag_y)) = state.maybe_drag_xy {
            let distance = f64::from(drag_x - x).hypot(f64::from(drag_y - y));
            if distance > f64::from(XY_CIRCLE_RADIUS) {
                state.maybe_drag_origin = Some((x, y));
            }
        } else {
            state.maybe_drag_origin = Some((x, y));
        }
        state.maybe_drag_xy = Some((x, y));
        state.dragging = true;
    }
}

/// Displays a UI for assignment the stickers of a `PuzzleGeometry`
///
/// # Errors
///
/// This function will return an `OpenCV` error.
pub fn pixel_assignment_ui(
    puzzle_geometry: &PuzzleGeometry,
) -> Result<Box<[Pixel]>, opencv::Error> {
    highgui::named_window(
        WINDOW_NAME,
        highgui::WINDOW_NORMAL | highgui::WINDOW_KEEPRATIO | highgui::WINDOW_GUI_EXPANDED,
    )?;

    let mut img = imgcodecs::imread_def("input.jpg")?;

    let w = img.cols();
    let h = img.rows();
    let mut pixel_count = w * h;

    if pixel_count > MAX_PIXEL_COUNT {
        let scale = (f64::from(MAX_PIXEL_COUNT) / f64::from(pixel_count)).sqrt();
        #[allow(clippy::cast_possible_truncation)]
        let new_w = (f64::from(w) * scale).round() as i32;
        #[allow(clippy::cast_possible_truncation)]
        let new_h = (f64::from(h) * scale).round() as i32;
        pixel_count = new_w * new_h;
        let mut resized = Mat::default();
        imgproc::resize(
            &img,
            &mut resized,
            Size::new(new_w, new_h),
            0.0,
            0.0,
            imgproc::INTER_AREA, // best for downscaling
        )?;
        img = resized;
    }

    let displayed_img = Mat::zeros(img.rows(), img.cols(), CV_8UC3)?.to_mat()?;
    let grayscale_mask = Mat::zeros(img.rows() + 2, img.cols() + 2, CV_8UC1)?.to_mat()?;
    let cleaned_grayscale_mask = grayscale_mask.clone();
    let eroded_grayscale_mask = grayscale_mask.clone();
    let tmp_mask = grayscale_mask.clone();
    let mask_roi = Rect::new(1, 1, img.cols(), img.rows());
    let erosion_kernel = Mat::default();
    let erosion_kernel_times_two = Mat::default();

    let pixel_assignment = vec![
        Pixel::Unassigned;
        pixel_count.try_into().map_err(|e| opencv::Error::new(
            opencv::core::StsError,
            format!("Too many pixels: {e}"),
        ))?
    ]
    .into_boxed_slice();

    let work = puzzle_geometry.stickers().to_vec();

    let state = Arc::new(Mutex::new(State {
        img,
        tmp_mask,
        grayscale_mask,
        cleaned_grayscale_mask,
        eroded_grayscale_mask,
        erosion_kernel,
        erosion_kernel_times_two,
        displayed_img,
        mask_roi,
        pixel_assignment,
        work,
        current_sticker_idx: 0,
        upper_flood_fill_diff: 0,
        maybe_drag_origin: None,
        maybe_drag_xy: None,
        maybe_xy: None,
        dragging: false,
        ui: UIState::Assigning,
    }));

    {
        let state = Arc::clone(&state);
        highgui::set_mouse_callback(
            WINDOW_NAME,
            Some(Box::new(move |event, x, y, _flags| {
                #[allow(clippy::missing_panics_doc)]
                let mut state = state.lock().unwrap();
                if let Err(e) = mouse_callback(&mut state, event, x, y) {
                    state.ui = UIState::OpenCVError(e);
                }
            })),
        )?;
    }
    {
        let state = Arc::clone(&state);
        highgui::create_trackbar(
            EROSION_SIZE_TRACKBAR_NAME,
            WINDOW_NAME,
            None,
            EROSION_SIZE_TRACKBAR_MINDEFMAX[2],
            Some(Box::new(move |pos| {
                #[allow(clippy::missing_panics_doc)]
                let mut state = state.lock().unwrap();
                if let Err(e) = erosion_kernel_trackbar_callback(&mut state, pos) {
                    state.ui = UIState::OpenCVError(e);
                }
            })),
        )?;
        highgui::set_trackbar_pos(
            EROSION_SIZE_TRACKBAR_NAME,
            WINDOW_NAME,
            EROSION_SIZE_TRACKBAR_MINDEFMAX[1],
        )?;
        highgui::set_trackbar_min(
            EROSION_SIZE_TRACKBAR_NAME,
            WINDOW_NAME,
            EROSION_SIZE_TRACKBAR_MINDEFMAX[0],
        )?;
    }
    {
        let state = Arc::clone(&state);
        highgui::create_trackbar(
            UPPER_DIFF_TRACKBAR_NAME,
            WINDOW_NAME,
            None,
            UPPER_DIFF_TRACKBAR_MINDEFMAX[2],
            Some(Box::new(move |pos| {
                #[allow(clippy::missing_panics_doc)]
                let mut state = state.lock().unwrap();
                if let Err(e) = light_tolerance_trackbar_callback(&mut state, pos) {
                    state.ui = UIState::OpenCVError(e);
                }
            })),
        )?;
        highgui::set_trackbar_pos(
            UPPER_DIFF_TRACKBAR_NAME,
            WINDOW_NAME,
            UPPER_DIFF_TRACKBAR_MINDEFMAX[1],
        )?;
        highgui::set_trackbar_min(
            UPPER_DIFF_TRACKBAR_NAME,
            WINDOW_NAME,
            UPPER_DIFF_TRACKBAR_MINDEFMAX[0],
        )?;
    }
    {
        let state = Arc::clone(&state);
        highgui::create_button_def(
            SUBMIT_BUTTON_NAME,
            Some(Box::new(move |_state| {
                #[allow(clippy::missing_panics_doc)]
                let mut state = state.lock().unwrap();
                if let Err(e) = submit_button_callback(&mut state) {
                    state.ui = UIState::OpenCVError(e);
                }
            })),
        )?;
    }

    {
        #[allow(clippy::missing_panics_doc)]
        let mut state = state.lock().unwrap();
        update_display(&mut state)?;
    }

    let mut in_toggle_dragging = false;
    loop {
        const D: i32 = 100;
        const R: i32 = 114;
        const S: i32 = 115;

        {
            #[allow(clippy::missing_panics_doc)]
            let state = state.lock().unwrap();
            match &state.ui {
                UIState::Finished => {
                    highgui::destroy_all_windows()?;
                    break Ok(state.pixel_assignment.clone());
                }
                UIState::OpenCVError(e) => {
                    highgui::destroy_all_windows()?;
                    break Err(opencv::Error::new(
                        e.code,
                        format!("OpenCV error during pixel assignment: {}", e.message),
                    ));
                }
                UIState::Assigning => (),
            }
        }

        let key = highgui::wait_key(1000 / 30)?;
        {
            #[allow(clippy::missing_panics_doc)]
            let mut state = state.lock().unwrap();
            match key {
                D => {
                    in_toggle_dragging = false;
                    submit_button_callback(&mut state)?;
                }
                R => {
                    in_toggle_dragging = false;
                    restart_button_callback(&mut state)?;
                }
                S => {
                    if !in_toggle_dragging {
                        toggle_dragging(&mut state);
                        in_toggle_dragging = true;
                    }
                }
                _ => {
                    in_toggle_dragging = false;
                }
            }
        }
    }
}
