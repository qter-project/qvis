use std::sync::Arc;

use internment::ArcIntern;
use puzzle_theory::{permutations::Permutation, puzzle_geometry::PuzzleGeometry};

use crate::{inference::Inference, puzzle_matching::Matcher};

mod inference;
pub mod puzzle_matching;

/// Processes images for computer vision
pub struct CVProcessor {
    image_size: usize,
    matcher: Matcher,
    inference: Inference,
}

pub enum Pixel {
    Unassigned,
    WhiteBalance(ArcIntern<str>),
    Sticker(usize),
}

impl CVProcessor {
    /// Create a new `CVProcessor` that recognizes the given puzzle in images. `image_size` specifies the number of pixels in the image. The CV algorithm does not care about rows and columns.
    ///
    /// # Assignment
    ///
    /// The assignment is the same size as the image.
    ///
    /// Each pixel is configured with a number determining which index sticker of the puzzle it belongs to. This method panics if any indices are out of range. The boolean parameter determines whether the pixel should be treated as white balance for the given face: `false` means that it is not white balance and `true` means that it is white balance.
    ///
    /// White balance points should be selected such that the face is parallel with the face that it is acting as white balance for.
    ///
    /// Pixels marked `None` will not be considered in the CV algorithm.
    pub fn new(
        puzzle: Arc<PuzzleGeometry>,
        image_size: usize,
        assignment: Box<[Pixel]>,
    ) -> CVProcessor {
        CVProcessor {
            image_size,
            inference: Inference::new(assignment, &puzzle),
            matcher: Matcher::new(puzzle),
        }
    }

    /// Calibrate the CV processor with an image of the puzzle in the given state.
    pub fn calibrate(&mut self, image: &[(f64, f64, f64)], state: Permutation) {
        assert_eq!(self.image_size, image.len());

        self.inference.calibrate(image, state);
    }

    /// Process an image and return the most likely state that the puzzle appears to be in, along with the confidence in the prediction. This is guaranteed to be a valid member of the group.
    pub fn process_image(&self, image: Box<[(f64, f64, f64)]>) -> (Permutation, f64) {
        self.matcher.most_likely(&self.inference.infer(&image))
    }
}
