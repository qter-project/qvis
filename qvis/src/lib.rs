use std::sync::{Arc, mpsc};

use puzzle_theory::{permutations::Permutation, puzzle_geometry::PuzzleGeometry};

mod puzzle_matching;

/// Processes images for computer vision
pub struct CVProcessor {
    puzzle: Arc<PuzzleGeometry>,
    image_size: usize,
}

impl CVProcessor {
    /// Create a new `CVProcessor` that recognizes the given puzzle in images. `image_size` specifies the number of pixels in the image. The CV algorithm does not care about rows and columns.
    pub fn new(puzzle: Arc<PuzzleGeometry>, image_size: usize) -> CVProcessor {
        CVProcessor { puzzle, image_size }
    }

    /// Calibrate the CV processor with an image of the puzzle in the given state.
    pub fn calibrate(&self, image: Box<[(f64, f64, f64)]>, state: Permutation) {
        assert_eq!(self.image_size, image.len());
        todo!()
    }

    /// Process an image and return the most likely state that the puzzle appears to be in, along with the confidence in the prediction. This is guaranteed to be a valid member of the group.
    ///
    /// To account for non-stationarities and new lighting conditions, it is recommended to call `calibrate` on the output.
    pub fn process_image(&self, image: Box<[(f64, f64, f64)]>) -> (Permutation, f64) {
        (Permutation::from_mapping(Vec::new()), 0.)
    }

    /// Set the mask of the image; the mask is the same size as the image.
    ///
    /// Each pixel is configured with a number determining which face of the puzzle it belongs to, such that pixels with the same number correspond to the same face. The boolean parameter determines whether the pixel should be treated as white balance for the given face: `false` means that it is not white balance and `true` means that it is white balance.
    ///
    /// White balance points should be selected such that the face is parallel with the face that it is acting as white balance for.
    ///
    /// Masking the image can give better CV results
    pub fn mask(&self, mask: Box<[Option<(u32, bool)>]>) {}
}
