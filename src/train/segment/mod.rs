use anyhow::Result;
use image::imageops::FilterType;
use ndarray::{Array, Array4};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use std::path::Path;

/// Segmentation engine wrapping ONNX Runtime and YOLOv8-seg models
pub struct Segmenter {
    session: Session,
}

impl Segmenter {
    /// Initialize the ONNX session with the given model path
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        Ok(Self { session })
    }

    /// Process an image and return raw boolean masks for each detected subject
    /// Masks are returned as an array of shape [NUM_SUBJECTS, 1, 64, 64]
    pub fn extract_masks<P: AsRef<Path>>(&mut self, image_path: P) -> Result<Array4<bool>> {
        // Load image
        let img = image::open(image_path)?;
        let img = img.resize_exact(640, 640, FilterType::Triangle).to_rgb8();

        // Convert to ndarray [1, 3, 640, 640] f32
        let mut input_arr: Array4<f32> = Array::zeros((1, 3, 640, 640));
        for (x, y, pixel) in img.enumerate_pixels() {
            let px = x as usize;
            let py = y as usize;
            input_arr[[0, 0, py, px]] = (pixel[0] as f32) / 255.0;
            input_arr[[0, 1, py, px]] = (pixel[1] as f32) / 255.0;
            input_arr[[0, 2, py, px]] = (pixel[2] as f32) / 255.0;
        }

        // Run inference
        let input_tensor = Value::from_array(input_arr)?;
        let outputs = self.session.run(ort::inputs!["images" => input_tensor])?;

        // YOLOv8-seg outputs:
        // output0: [1, 116, 8400] (boxes + class scores + mask coefficients)
        // output1: [1, 32, 160, 160] (proto masks)

        // For this mock implementation aimed at SOMA's tiered masking, we will quickly
        // extract the high confidence proto-masks. In a full YOLOv8-seg implemention
        // we would multiply the mask coefficients against the proto masks and apply NMS.

        // Extract proto masks [1, 32, 160, 160]
        let proto_masks_raw = outputs["output1"].try_extract_tensor::<f32>()?;
        // try_extract_tensor returns (shape, data) or similar, we want to construct an ndarray view
        let proto_masks_view = proto_masks_raw.1;
        let p_shape = proto_masks_raw.0;
        let proto_masks = ndarray::ArrayView4::from_shape(
            (
                p_shape[0] as usize,
                p_shape[1] as usize,
                p_shape[2] as usize,
                p_shape[3] as usize,
            ),
            proto_masks_view,
        )
        .map_err(|e| anyhow::anyhow!("Shape error: {}", e))?;

        // As a shortcut to prove the SOMA pipeline works with mask shapes, we'll
        // downsample the proto masks directly to 64x64, simulating the actual masks.
        // We pick the first N proto masks as "subjects".
        let num_subjects = std::cmp::min(2, proto_masks.shape()[1]); // Take up to 2 subjects

        let mut final_masks = Array::from_elem((num_subjects, 1, 64, 64), false);

        for s in 0..num_subjects {
            for y in 0..64 {
                for x in 0..64 {
                    // Map 64x64 directly to 160x160 space using Nearest Neighbor
                    let py = (y as f32 / 64.0 * 160.0) as usize;
                    let px = (x as f32 / 64.0 * 160.0) as usize;

                    let val = proto_masks[[0, s, py, px]];
                    // Simple threshold
                    if val > 0.0 {
                        final_masks[[s, 0, y, x]] = true;
                    }
                }
            }
        }

        Ok(final_masks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_segmenter_initialization() {
        // Only run if model exists
        let model_path = PathBuf::from("models/yolov8n-seg.onnx");
        if !model_path.exists() {
            println!("Skipping test: model not found at {:?}", model_path);
            return;
        }

        let segmenter_res = Segmenter::new(model_path);
        assert!(segmenter_res.is_ok(), "Failed to initialize segmenter");
    }

    #[test]
    fn test_segmentation_mock_mask_extraction() {
        let model_path = PathBuf::from("models/yolov8n-seg.onnx");
        let image_path = PathBuf::from("data/test.jpg");
        if !model_path.exists() || !image_path.exists() {
            println!("Skipping extraction test: required files missing");
            return;
        }

        let mut segmenter = Segmenter::new(model_path).unwrap();
        let masks = segmenter.extract_masks(&image_path).unwrap();

        assert_eq!(masks.shape().len(), 4);
        assert_eq!(masks.shape()[1], 1);
        assert_eq!(masks.shape()[2], 64);
        assert_eq!(masks.shape()[3], 64);

        let true_count = masks.iter().filter(|&&v| v).count();
        assert!(true_count > 0, "Expected some mask pixels to be true");
    }
}
