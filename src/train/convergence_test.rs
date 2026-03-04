use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::prelude::*;
use burn::tensor::Distribution;

use crate::train::loss::flow_matching::{
    compute_target_velocity, interpolate_latent, velocity_prediction_loss,
};

type B = Autodiff<NdArray>;

#[derive(Module, Debug)]
struct DummyVelocityPredictor<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
}

impl<B: Backend> DummyVelocityPredictor<B> {
    fn new(device: &B::Device) -> Self {
        let conv1 = Conv2dConfig::new([4, 16], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let conv2 = Conv2dConfig::new([16, 4], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        Self { conv1, conv2 }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = self.conv1.forward(x);
        self.conv2.forward(h)
    }
}

#[test]
fn dummy_model_loss_decreases_over_steps() {
    let device = <B as Backend>::Device::default();
    let mut model = DummyVelocityPredictor::<B>::new(&device);
    let mut optim = SgdConfig::new().init::<B, DummyVelocityPredictor<B>>();
    let lr = 0.01_f64;

    let x0: Tensor<NdArray, 4> =
        Tensor::random([1, 4, 8, 8], Distribution::Normal(0.0, 1.0), &device);
    let x1: Tensor<NdArray, 4> =
        Tensor::random([1, 4, 8, 8], Distribution::Normal(0.0, 1.0), &device);
    let target_velocity = compute_target_velocity(x0.clone(), x1.clone());
    let xt = interpolate_latent(x0, x1, 0.5);

    let mut first_loss_value = f32::MAX;
    let mut last_loss_value = f32::MAX;

    for step in 0..50 {
        let xt_ad: Tensor<B, 4> = Tensor::from_data(xt.to_data(), &device);
        let target_ad: Tensor<B, 4> = Tensor::from_data(target_velocity.to_data(), &device);

        let predicted = model.forward(xt_ad);
        let loss = velocity_prediction_loss(predicted, target_ad);

        let loss_value: f32 = loss.clone().into_data().to_vec::<f32>().unwrap()[0];

        if step == 0 {
            first_loss_value = loss_value;
        }
        last_loss_value = loss_value;

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);
    }

    assert!(
        last_loss_value < first_loss_value,
        "loss should decrease: first={first_loss_value}, last={last_loss_value}"
    );

    let reduction_pct = (1.0 - last_loss_value / first_loss_value) * 100.0;
    assert!(
        reduction_pct > 10.0,
        "loss should decrease by at least 10%, got {reduction_pct:.1}%"
    );
}
