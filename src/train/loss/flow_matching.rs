use burn::tensor::{backend::Backend, Tensor};

pub fn velocity_prediction_loss<B: Backend>(
    predicted_velocity: Tensor<B, 4>,
    target_velocity: Tensor<B, 4>,
) -> Tensor<B, 1> {
    let diff = predicted_velocity.sub(target_velocity);
    let squared_error = diff.clone().mul(diff);
    squared_error.mean()
}

pub fn compute_target_velocity<B: Backend>(x0: Tensor<B, 4>, x1: Tensor<B, 4>) -> Tensor<B, 4> {
    x1.sub(x0)
}

pub fn interpolate_latent<B: Backend>(x0: Tensor<B, 4>, x1: Tensor<B, 4>, t: f32) -> Tensor<B, 4> {
    let scaled_noise = x0.clone().mul_scalar(1.0 - t);
    let scaled_data = x1.clone().mul_scalar(t);
    scaled_noise.add(scaled_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    use burn::backend::Autodiff;
    use burn::tensor::Distribution;

    type TestBackend = NdArray;
    type TestAutodiffBackend = Autodiff<TestBackend>;

    #[test]
    fn loss_is_zero_when_prediction_matches_target() {
        let device = <TestBackend as Backend>::Device::default();
        let target: Tensor<TestBackend, 4> =
            Tensor::random([1, 4, 8, 8], Distribution::Normal(0.0, 1.0), &device);
        let predicted = target.clone();

        let loss = velocity_prediction_loss(predicted, target);

        let loss_value = loss.into_scalar();
        assert!(
            loss_value.abs() < 1e-6,
            "loss should be zero when prediction matches target, got {loss_value}"
        );
    }

    #[test]
    fn loss_is_positive_when_prediction_differs() {
        let device = <TestBackend as Backend>::Device::default();
        let target: Tensor<TestBackend, 4> =
            Tensor::random([1, 4, 8, 8], Distribution::Normal(0.0, 1.0), &device);
        let predicted: Tensor<TestBackend, 4> =
            Tensor::random([1, 4, 8, 8], Distribution::Normal(0.0, 1.0), &device);

        let loss = velocity_prediction_loss(predicted, target);

        let loss_value = loss.into_scalar();
        assert!(
            loss_value > 0.0,
            "loss should be positive for different tensors, got {loss_value}"
        );
    }

    #[test]
    fn target_velocity_is_data_minus_noise() {
        let device = <TestBackend as Backend>::Device::default();
        let x0: Tensor<TestBackend, 4> =
            Tensor::random([1, 4, 8, 8], Distribution::Normal(0.0, 1.0), &device);
        let x1: Tensor<TestBackend, 4> =
            Tensor::random([1, 4, 8, 8], Distribution::Normal(0.0, 1.0), &device);

        let velocity = compute_target_velocity(x0.clone(), x1.clone());
        let expected = x1.sub(x0);
        let diff = velocity.sub(expected);
        let max_err: f32 = diff.abs().max().into_scalar();

        assert!(
            max_err < 1e-6,
            "target velocity should equal x1 - x0, max error: {max_err}"
        );
    }

    #[test]
    fn interpolation_boundaries() {
        let device = <TestBackend as Backend>::Device::default();
        let x0: Tensor<TestBackend, 4> =
            Tensor::random([1, 4, 8, 8], Distribution::Normal(0.0, 1.0), &device);
        let x1: Tensor<TestBackend, 4> =
            Tensor::random([1, 4, 8, 8], Distribution::Normal(0.5, 1.0), &device);

        let at_zero = interpolate_latent(x0.clone(), x1.clone(), 0.0);
        let diff_zero = at_zero.sub(x0.clone()).abs().max().into_scalar();
        assert!(
            diff_zero < 1e-6,
            "interpolation at t=0 should equal x0, diff: {diff_zero}"
        );

        let at_one = interpolate_latent(x0, x1.clone(), 1.0);
        let diff_one: f32 = at_one.sub(x1).abs().max().into_scalar();
        assert!(
            diff_one < 1e-6,
            "interpolation at t=1 should equal x1, diff: {diff_one}"
        );
    }

    #[test]
    fn loss_produces_gradients() {
        let device = <TestAutodiffBackend as Backend>::Device::default();

        let predicted: Tensor<TestAutodiffBackend, 4> =
            Tensor::random([1, 4, 8, 8], Distribution::Normal(0.0, 1.0), &device).require_grad();

        let target: Tensor<TestAutodiffBackend, 4> =
            Tensor::random([1, 4, 8, 8], Distribution::Normal(0.0, 1.0), &device);

        let loss = velocity_prediction_loss(predicted.clone(), target);
        let grads = loss.backward();
        let pred_grad = predicted.grad(&grads);

        assert!(
            pred_grad.is_some(),
            "gradients should exist for the predicted tensor after backward pass"
        );

        let grad_tensor = pred_grad.unwrap();
        let grad_sum: f32 = grad_tensor.abs().sum().into_scalar();
        assert!(
            grad_sum > 0.0,
            "gradient magnitude should be non-zero, got {grad_sum}"
        );
    }
}
