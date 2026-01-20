use opencv::{
    core::{BORDER_CONSTANT, CV_8UC1, Point, Rect, Scalar, Size},
    highgui, imgcodecs,
    imgproc::{self, FLOODFILL_FIXED_RANGE, FLOODFILL_MASK_ONLY},
    prelude::*,
};
use std::{
    f64::consts::PI,
    sync::{Arc, Mutex},
};

const WINDOW_NAME: &str = "Qvis Sticker Calibration";
const ERODE_DEF_ANCHOR: Point = Point::new(-1, -1);

struct State {
    img: Mat,
    grayscale_mask: Mat,
    eroded_mask: Mat,
    erosion_kernel: Mat,
    mask_roi: Rect,
    displayed_img: Mat,
    upper_flood_fill_diff: i32,
    maybe_drag_xy: Option<(i32, i32)>,
    err: Option<opencv::Error>,
}

fn c(x: i32, n: i32) -> i32 {
    (x + n).checked_div_euclid(6).unwrap()
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
            let Some((drag_x, drag_y)) = state.maybe_drag_xy else {
                return Ok(());
            };
            let distance = ((x - drag_x) as f64).hypot((y - drag_y) as f64) as i32;
            let angle = (((y - drag_y) as f64).atan2((x - drag_x) as f64) * 1440.0 / PI) as u16;
            let perm6 = perm6_from_number(angle);

            state.grayscale_mask.set_to_def(&Scalar::all(0.0))?;
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
                    c(distance, perm6[3] + state.upper_flood_fill_diff),
                    c(distance, perm6[4] + state.upper_flood_fill_diff),
                    c(distance, perm6[5] + state.upper_flood_fill_diff),
                )),
                4 | FLOODFILL_FIXED_RANGE | FLOODFILL_MASK_ONLY | (255 << 8),
            )?;
            imgproc::erode(
                &state.grayscale_mask,
                &mut state.eroded_mask,
                &state.erosion_kernel,
                ERODE_DEF_ANCHOR,
                2,
                BORDER_CONSTANT,
                imgproc::morphology_default_border_value()?,
            )?;
            *state.eroded_mask.at_2d_mut::<u8>(drag_y + 1, drag_x + 1)? = 255;
            {
                let tmp_mask = &mut state.grayscale_mask;
                tmp_mask.set_to_def(&Scalar::all(0.0))?;
                imgproc::flood_fill_mask(
                    &mut Mat::roi_mut(&mut state.eroded_mask, state.mask_roi)?,
                    tmp_mask,
                    Point::new(drag_x, drag_y),
                    Scalar::default(), // ignored
                    &mut Rect::default(),
                    Scalar::all(0.0),
                    Scalar::all(0.0),
                    4 | FLOODFILL_FIXED_RANGE | FLOODFILL_MASK_ONLY | (255 << 8),
                )?;
                std::mem::swap(&mut state.eroded_mask, tmp_mask);
            }
            let eroded_mask_cropped = Mat::roi(&state.eroded_mask, state.mask_roi)?;
            let channels = opencv::core::Vector::<Mat>::from_iter([
                eroded_mask_cropped.clone_pointee(),
                Mat::zeros(state.img.rows(), state.img.cols(), CV_8UC1)?.to_mat()?,
                eroded_mask_cropped.clone_pointee(),
            ]);
            opencv::core::merge(&channels, &mut state.eroded_mask)?;

            opencv::core::add_def(&state.img, &state.eroded_mask, &mut state.displayed_img)?;
            highgui::imshow(WINDOW_NAME, &state.displayed_img)?;
        }
        highgui::EVENT_LBUTTONDOWN => {
            state.maybe_drag_xy = Some((x, y));
        }
        highgui::EVENT_LBUTTONUP => {
            state.maybe_drag_xy = None;
        }
        _ => (),
    }

    Ok(())
}

fn main() -> opencv::Result<()> {
    highgui::named_window(WINDOW_NAME, highgui::WINDOW_AUTOSIZE)?;

    let img = imgcodecs::imread("input.png", imgcodecs::IMREAD_COLOR)?;
    let displayed_img = Mat::zeros(img.rows(), img.cols(), img.typ())?.to_mat()?;
    let grayscale_mask = Mat::zeros(img.rows() + 2, img.cols() + 2, CV_8UC1)?.to_mat()?;
    let eroded_mask = grayscale_mask.clone();
    let mask_roi = Rect::new(1, 1, img.cols(), img.rows());
    let erosion_kernel =
        imgproc::get_structuring_element_def(imgproc::MORPH_DIAMOND, Size::new(5, 5))?;

    highgui::imshow(WINDOW_NAME, &img)?;

    let state = Arc::new(Mutex::new(State {
        img,
        grayscale_mask,
        eroded_mask,
        erosion_kernel,
        mask_roi,
        displayed_img,
        upper_flood_fill_diff: 20,
        maybe_drag_xy: None,
        err: None,
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
