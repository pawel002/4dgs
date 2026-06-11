from argparse import ArgumentParser

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams

parser = ArgumentParser(description="Testing script parameters")
params = ModelParams(parser)
params._model_path = "output/1c3a151f-c/point_cloud/iteration_7000/point_cloud.ply"

scene = Scene(params, GaussianModel())