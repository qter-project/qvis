use opencv::{
    core::{BORDER_CONSTANT, CV_8UC1, CV_8UC3, Point, Rect, Scalar, Size},
    highgui, imgcodecs,
    imgproc::{self, FILLED, FLOODFILL_FIXED_RANGE, FLOODFILL_MASK_ONLY, LINE_8},
    prelude::*,
};
use std::{
    f64::consts::PI,
    sync::{Arc, Mutex},
};

const WINDOW_NAME: &str = "Qvis Sticker Calibration";
const EROSION_SIZE_TRACKBAR_NAME: &str = "Erosion size";
const EROSION_SIZE_TRACKBAR_MINDEFMAX: [i32; 3] = [1, 5, 30];
const UPPER_DIFF_TRACKBAR_NAME: &str = "Upper diff";
const UPPER_DIFF_TRACKBAR_MINDEFMAX: [i32; 3] = [0, 2, 5];
const ERODE_DEF_ANCHOR: Point = Point::new(-1, -1);
const XY_CIRCLE_RADIUS: i32 = 10;
const MAX_PIXEL: i32 = 255;

#[derive(Default)]
struct State {
    img: Mat,
    tmp_mask: Mat,
    grayscale_mask: Mat,
    eroded_grayscale_mask: Mat,
    colored_eroded_mask_cropped: Mat,
    erosion_kernel: Mat,
    erosion_kernel_times_two: Mat,
    displayed_img: Mat,
    mask_roi: Rect,
    upper_flood_fill_diff: i32,

    maybe_drag_origin: Option<(i32, i32)>,
    maybe_xy: Option<(i32, i32)>,
    dragging: bool,
    err: Option<opencv::Error>,
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

fn mouse_callback(state: &mut State, event: i32, x: i32, y: i32) -> opencv::Result<()> {
    match event {
        highgui::EVENT_MOUSEMOVE => {
            if !state.dragging {
                return Ok(());
            }
            state.maybe_xy = Some((x, y));

            overlay_flood_fill(state)?;
        }
        highgui::EVENT_LBUTTONDOWN => {
            if let Some((old_x, old_y)) = state.maybe_xy {
                let distance = ((old_x - x) as f64).hypot((old_y - y) as f64) as i32;
                if distance > XY_CIRCLE_RADIUS {
                    state.maybe_drag_origin = Some((x, y));
                }
            } else {
                state.maybe_drag_origin = Some((x, y));
            }
            state.maybe_xy = Some((x, y));
            state.dragging = true;
        }
        highgui::EVENT_LBUTTONUP => {
            state.dragging = false;
        }
        _ => (),
    }

    Ok(())
}

fn overlay_flood_fill(state: &mut State) -> opencv::Result<()> {
    let Some((drag_x, drag_y)) = state.maybe_drag_origin else {
        return Ok(());
    };
    let Some((x, y)) = state.maybe_xy else {
        return Ok(());
    };

    let distance = ((x - drag_x) as f64).hypot((y - drag_y) as f64) as i32 / 4;
    // angle is between [-pi/2, pi/2]; find the absolute value and
    // multiply by 1440/pi to get a range of [0, 720] throughout the
    // full circle which is 6!
    //
    // multiply it again by 20 to increase the periodicity
    let angle =
        (((y - drag_y) as f64).atan2((x - drag_x) as f64).abs() * 1440.0 / PI * 20.0) as u16;
    let perm6 = perm6_from_number(angle);

    state.grayscale_mask.set_to_def(&Scalar::all(0.0))?; // needed
    imgproc::flood_fill_mask(
        &mut state.img,
        &mut state.grayscale_mask,
        Point::new(drag_x, drag_y),
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
                perm6[3] + state.upper_flood_fill_diff * MAX_PIXEL / UPPER_DIFF_TRACKBAR_MINDEFMAX[2],
            ),
            c(
                distance,
                perm6[4] + state.upper_flood_fill_diff * MAX_PIXEL / UPPER_DIFF_TRACKBAR_MINDEFMAX[2],
            ),
            c(
                distance,
                perm6[5] + state.upper_flood_fill_diff * MAX_PIXEL / UPPER_DIFF_TRACKBAR_MINDEFMAX[2],
            ),
        )),
        4 | FLOODFILL_FIXED_RANGE | FLOODFILL_MASK_ONLY | (MAX_PIXEL << 8),
    )?;

    imgproc::erode(
        &state.grayscale_mask,
        &mut state.eroded_grayscale_mask,
        &state.erosion_kernel,
        ERODE_DEF_ANCHOR,
        2,
        BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;
    let to_dilate = if opencv::core::has_non_zero(&Mat::roi(&state.eroded_grayscale_mask, state.mask_roi)?)? {
        *state
            .eroded_grayscale_mask
            .at_2d_mut::<u8>(drag_y + 1, drag_x + 1)? = MAX_PIXEL as u8;

        state.tmp_mask.set_to_def(&Scalar::all(0.0))?;
        imgproc::flood_fill_mask(
            &mut Mat::roi_mut(&mut state.eroded_grayscale_mask, state.mask_roi)?,
            &mut state.tmp_mask,
            Point::new(drag_x, drag_y),
            Scalar::default(), // ignored
            &mut Rect::default(),
            Scalar::all(0.0),
            Scalar::all(0.0),
            4 | FLOODFILL_FIXED_RANGE | FLOODFILL_MASK_ONLY | (MAX_PIXEL << 8),
        )?;
        std::mem::swap(&mut state.eroded_grayscale_mask, &mut state.tmp_mask);
        &state.eroded_grayscale_mask
    } else {
        &state.grayscale_mask
    };

    imgproc::dilate(
        to_dilate,
        &mut state.tmp_mask,
        &state.erosion_kernel_times_two,
        ERODE_DEF_ANCHOR,
        1,
        BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;
    std::mem::swap(&mut state.eroded_grayscale_mask, &mut state.tmp_mask);

    let eroded_grayscale_mask_cropped = Mat::roi(&state.eroded_grayscale_mask, state.mask_roi)?;
    let colored_eroded_mask_cropped_channels = opencv::core::Vector::<Mat>::from_iter([
        eroded_grayscale_mask_cropped.clone_pointee(),
        Mat::zeros(state.img.rows(), state.img.cols(), CV_8UC1)?.to_mat()?,
        eroded_grayscale_mask_cropped.clone_pointee(),
    ]);
    opencv::core::merge(
        &colored_eroded_mask_cropped_channels,
        &mut state.colored_eroded_mask_cropped,
    )?;

    opencv::core::add_def(
        &state.img,
        &state.colored_eroded_mask_cropped,
        &mut state.displayed_img,
    )?;
    imgproc::line(
        &mut state.displayed_img,
        Point::new(drag_x, drag_y),
        Point::new(x, y),
        Scalar::all(MAX_PIXEL as f64),
        3,
        LINE_8,
        0,
    )?;
    imgproc::circle(
        &mut state.displayed_img,
        Point::new(x, y),
        XY_CIRCLE_RADIUS,
        Scalar::all(MAX_PIXEL as f64),
        FILLED,
        LINE_8,
        0,
    )?;
    imgproc::circle(
        &mut state.displayed_img,
        Point::new(x, y),
        XY_CIRCLE_RADIUS / 2,
        Scalar::from((0, 0, MAX_PIXEL)),
        FILLED,
        LINE_8,
        0,
    )?;
    highgui::imshow(WINDOW_NAME, &state.displayed_img)?;
    Ok(())
}

fn erosion_kernel_trackbar_callback(state: &mut State, pos: i32) -> opencv::Result<()> {
    state.erosion_kernel =
        imgproc::get_structuring_element_def(imgproc::MORPH_CROSS, Size::new(pos, pos))?;
    state.erosion_kernel_times_two =
        imgproc::get_structuring_element_def(imgproc::MORPH_CROSS, Size::new(pos * 2, pos * 2))?;
    overlay_flood_fill(state)?;
    Ok(())
}

fn light_tolerance_trackbar_callback(state: &mut State, pos: i32) -> opencv::Result<()> {
    state.upper_flood_fill_diff = pos;
    overlay_flood_fill(state)?;
    Ok(())
}

fn main() -> opencv::Result<()> {
    highgui::named_window(WINDOW_NAME, highgui::WINDOW_AUTOSIZE)?;

    let img = imgcodecs::imread("input.png", imgcodecs::IMREAD_COLOR)?;
    let displayed_img = Mat::zeros(img.rows(), img.cols(), CV_8UC3)?.to_mat()?;
    let colored_eroded_mask_cropped = displayed_img.clone();
    let grayscale_mask = Mat::zeros(img.rows() + 2, img.cols() + 2, CV_8UC1)?.to_mat()?;
    let eroded_grayscale_mask = grayscale_mask.clone();
    let tmp_mask = grayscale_mask.clone();
    let mask_roi = Rect::new(1, 1, img.cols(), img.rows());
    let erosion_kernel = Mat::default();
    let erosion_kernel_times_two = Mat::default();

    let state = Arc::new(Mutex::new(State {
        img,
        grayscale_mask,
        eroded_grayscale_mask,
        erosion_kernel,
        erosion_kernel_times_two,
        colored_eroded_mask_cropped,
        mask_roi,
        tmp_mask,
        displayed_img,
        ..Default::default()
    }));

    {
        let state = Arc::clone(&state);
        highgui::set_mouse_callback(
            WINDOW_NAME,
            Some(Box::new(move |event, x, y, _flags| {
                let mut state = state.lock().unwrap();
                if let Err(e) = mouse_callback(&mut state, event, x, y) {
                    state.err = Some(e);
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
                let mut state = state.lock().unwrap();
                if let Err(e) = erosion_kernel_trackbar_callback(&mut state, pos) {
                    state.err = Some(e);
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
                let mut state = state.lock().unwrap();
                if let Err(e) = light_tolerance_trackbar_callback(&mut state, pos) {
                    state.err = Some(e);
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

    highgui::imshow(WINDOW_NAME, &state.lock().unwrap().img)?;

    loop {
        if let Some(err) = state.lock().unwrap().err.take() {
            return Err(err);
        }
        if highgui::wait_key(1000 / 30)? == 27 {
            break;
        }
    }

    Ok(())
}
