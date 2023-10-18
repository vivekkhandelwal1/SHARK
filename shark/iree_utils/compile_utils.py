# Copyright 2023 The Nod Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
import numpy as np
import os
import re
import tempfile
import time
from pathlib import Path

import iree.runtime as ireert
import iree.compiler as ireec
from shark.parser import shark_args

from .trace import DetailLogger
from ._common import iree_device_map, iree_target_map
from .cpu_utils import get_iree_cpu_rt_args
from .benchmark_utils import *


# Get the iree-compile arguments given device.
def get_iree_device_args(device, extra_args=[]):
    print("Configuring for device:" + device)
    device_uri = device.split("://")
    if len(device_uri) > 1:
        if device_uri[0] not in ["vulkan"]:
            print(
                f"Specific device selection only supported for vulkan now."
                f"Proceeding with {device} as device."
            )
        device_num = device_uri[1]
    else:
        device_num = 0

    if "cpu" in device:
        from shark.iree_utils.cpu_utils import get_iree_cpu_args

        data_tiling_flag = ["--iree-opt-data-tiling"]
        u_kernel_flag = ["--iree-llvmcpu-enable-microkernels"]
        stack_size_flag = ["--iree-llvmcpu-stack-allocation-limit=256000"]

        return (
            get_iree_cpu_args()
            + data_tiling_flag
            + u_kernel_flag
            + stack_size_flag
            # + ["--iree-flow-enable-quantized-matmul-reassociation"]
            # + ["--iree-llvmcpu-enable-quantized-matmul-reassociation"]
        )
    if device_uri[0] == "cuda":
        from shark.iree_utils.gpu_utils import get_iree_gpu_args

        return get_iree_gpu_args()
    if device_uri[0] == "vulkan":
        from shark.iree_utils.vulkan_utils import get_iree_vulkan_args

        return get_iree_vulkan_args(
            device_num=device_num, extra_args=extra_args
        )
    if device_uri[0] == "metal":
        from shark.iree_utils.metal_utils import get_iree_metal_args

        return get_iree_metal_args(extra_args=extra_args)
    if device_uri[0] == "rocm":
        from shark.iree_utils.gpu_utils import get_iree_rocm_args

        return get_iree_rocm_args()
    return []


# Get the iree-compiler arguments given frontend.
def get_iree_frontend_args(frontend):
    if frontend in ["torch", "pytorch", "linalg", "tm_tensor"]:
        return ["--iree-llvmcpu-target-cpu-features=host"]
    elif frontend in ["tensorflow", "tf", "mhlo", "stablehlo"]:
        return [
            "--iree-llvmcpu-target-cpu-features=host",
            "--iree-input-demote-i64-to-i32",
        ]
    else:
        # Frontend not found.
        return []


# Common args to be used given any frontend or device.
def get_iree_common_args(debug=False):
    common_args = [
        "--iree-stream-resource-max-allocation-size=4294967295",
        "--iree-vm-bytecode-module-strip-source-map=true",
        "--iree-util-zero-fill-elided-attrs",
    ]
    if debug == True:
        common_args.extend(
            [
                "--iree-opt-strip-assertions=false",
                "--verify=true",
            ]
        )
    else:
        common_args.extend(
            [
                "--iree-opt-strip-assertions=true",
                "--verify=false",
            ]
        )
    return common_args


# Args that are suitable only for certain models or groups of models.
# shark_args are passed down from pytests to control which models compile with these flags,
# but they can also be set in shark/parser.py
def get_model_specific_args():
    ms_args = []
    if shark_args.enable_conv_transform == True:
        ms_args += [
            "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-flow-convert-conv-nchw-to-nhwc))"
        ]
    if shark_args.enable_img2col_transform == True:
        ms_args += [
            "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-preprocessing-convert-conv2d-to-img2col))"
        ]
    if shark_args.use_winograd == True:
        ms_args += [
            "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-linalg-ext-convert-conv2d-to-winograd))"
        ]
    return ms_args


def create_dispatch_dirs(bench_dir, device):
    protected_files = ["ordered-dispatches.txt"]
    bench_dir_path = bench_dir.split("/")
    bench_dir_path[-1] = "temp_" + bench_dir_path[-1]
    tmp_bench_dir = "/".join(bench_dir_path)
    for f_ in os.listdir(bench_dir):
        if os.path.isfile(f"{bench_dir}/{f_}") and f_ not in protected_files:
            dir_name = re.sub("\.\S*$", "", f_)
            if os.path.exists(f"{bench_dir}/{dir_name}"):
                os.system(f"rm -rf {bench_dir}/{dir_name}")
            os.system(f"mkdir {bench_dir}/{dir_name}")
            os.system(f"mv {bench_dir}/{f_} {bench_dir}/{dir_name}/{f_}")
    for f_ in os.listdir(tmp_bench_dir):
        if os.path.isfile(f"{tmp_bench_dir}/{f_}"):
            dir_name = ""
            for d_ in os.listdir(bench_dir):
                if re.search(f"{d_}(?=\D)", f_):
                    dir_name = d_
            if dir_name != "":
                os.system(
                    f"mv {tmp_bench_dir}/{f_} {bench_dir}/{dir_name}/{dir_name}_benchmark.mlir"
                )


def dump_isas(bench_dir):
    for d_ in os.listdir(bench_dir):
        if os.path.isdir(f"{bench_dir}/{d_}"):
            for f_ in os.listdir(f"{bench_dir}/{d_}"):
                if f_.endswith(".spv"):
                    os.system(
                        f"amdllpc -gfxip 11.0 {bench_dir}/{d_}/{f_} -v > \
                         {bench_dir}/{d_}/isa.txt"
                    )


def compile_benchmark_dirs(bench_dir, device, dispatch_benchmarks):
    benchmark_runtimes = {}
    dispatch_list = []
    all_dispatches = False

    if dispatch_benchmarks.lower().strip() == "all":
        all_dispatches = True
    else:
        try:
            dispatch_list = [
                int(dispatch_index)
                for dispatch_index in dispatch_benchmarks.split(" ")
            ]
        except:
            print("ERROR: Invalid dispatch benchmarks")
            return None
    for d_ in os.listdir(bench_dir):
        if os.path.isdir(f"{bench_dir}/{d_}"):
            in_dispatches = False
            for dispatch in dispatch_list:
                if str(dispatch) in d_:
                    in_dispatches = True
            if all_dispatches or in_dispatches:
                for f_ in os.listdir(f"{bench_dir}/{d_}"):
                    if "benchmark.mlir" in f_:
                        dispatch_file = open(f"{bench_dir}/{d_}/{f_}", "r")
                        module = dispatch_file.read()
                        dispatch_file.close()

                        flatbuffer_blob = ireec.compile_str(
                            module, target_backends=[iree_target_map(device)]
                        )

                        vmfb_file = open(
                            f"{bench_dir}/{d_}/{d_}_benchmark.vmfb", "wb"
                        )
                        vmfb_file.write(flatbuffer_blob)
                        vmfb_file.close()

                        config = get_iree_runtime_config(device)
                        vm_module = ireert.VmModule.from_buffer(
                            config.vm_instance,
                            flatbuffer_blob,
                            warn_if_copy=False,
                        )

                        benchmark_cl = build_benchmark_args_non_tensor_input(
                            input_file=f"{bench_dir}/{d_}/{d_}_benchmark.vmfb",
                            device=device,
                            inputs=(0,),
                            mlir_dialect="linalg",
                            function_name="",
                        )

                        benchmark_bash = open(
                            f"{bench_dir}/{d_}/{d_}_benchmark.sh", "w+"
                        )
                        benchmark_bash.write("#!/bin/bash\n")
                        benchmark_bash.write(" ".join(benchmark_cl))
                        benchmark_bash.close()

                        iter_per_second, _, _ = run_benchmark_module(
                            benchmark_cl
                        )

                        benchmark_file = open(
                            f"{bench_dir}/{d_}/{d_}_data.txt", "w+"
                        )
                        benchmark_file.write(f"DISPATCH: {d_}\n")
                        benchmark_file.write(str(iter_per_second) + "\n")
                        benchmark_file.write(
                            "SHARK BENCHMARK RESULT: "
                            + str(1 / (iter_per_second * 0.001))
                            + "\n"
                        )
                        benchmark_file.close()

                        benchmark_runtimes[d_] = 1 / (iter_per_second * 0.001)

                    elif ".mlir" in f_ and "benchmark" not in f_:
                        dispatch_file = open(f"{bench_dir}/{d_}/{f_}", "r")
                        module = dispatch_file.read()
                        dispatch_file.close()

                        module = re.sub(
                            "hal.executable private",
                            "hal.executable public",
                            module,
                        )

                        flatbuffer_blob = ireec.compile_str(
                            module,
                            target_backends=[iree_target_map(device)],
                            extra_args=["--compile-mode=hal-executable"],
                        )

                        spirv_file = open(
                            f"{bench_dir}/{d_}/{d_}_spirv.vmfb", "wb"
                        )
                        spirv_file.write(flatbuffer_blob)
                        spirv_file.close()

    ordered_dispatches = [
        (k, v)
        for k, v in sorted(
            benchmark_runtimes.items(), key=lambda item: item[1]
        )
    ][::-1]
    f_ = open(f"{bench_dir}/ordered-dispatches.txt", "w+")
    for dispatch in ordered_dispatches:
        f_.write(f"{dispatch[0]}: {dispatch[1]}ms\n")
    f_.close()


def compile_module_to_flatbuffer(
    module,
    device,
    frontend,
    model_config_path,
    extra_args,
    model_name="None",
    debug=False,
):
    # Setup Compile arguments wrt to frontends.
    input_type = ""
    args = get_iree_frontend_args(frontend)
    args += get_iree_device_args(device, extra_args)
    args += get_iree_common_args(debug=debug)
    args += get_model_specific_args()
    args += extra_args
    args += shark_args.additional_compile_args

    if frontend in ["tensorflow", "tf"]:
        input_type = "auto"
    elif frontend in ["stablehlo", "tosa"]:
        input_type = frontend
    elif frontend in ["tflite", "tflite-tosa"]:
        input_type = "tosa"
    elif frontend in ["tm_tensor"]:
        input_type = ireec.InputType.TM_TENSOR

    # TODO: make it simpler.
    # Compile according to the input type, else just try compiling.
    if input_type != "":
        # Currently for MHLO/TOSA.
        flatbuffer_blob = ireec.compile_str(
            module,
            target_backends=[iree_target_map(device)],
            extra_args=args,
            input_type=input_type,
        )
    else:
        # Currently for Torch.
        flatbuffer_blob = ireec.compile_str(
            module,
            target_backends=[iree_target_map(device)],
            extra_args=args,
        )

    return flatbuffer_blob


def get_iree_module(flatbuffer_blob, device, device_idx=None):
    # Returns the compiled module and the configs.
    if device_idx is not None:
        device = iree_device_map(device)
        print("registering device id: ", device_idx)
        haldriver = ireert.get_driver(device)
        haldevice = haldriver.create_device(
            haldriver.query_available_devices()[device_idx]["device_id"],
            allocators=shark_args.device_allocator,
        )
        config = ireert.Config(device=haldevice)
    else:
        config = get_iree_runtime_config(device)
    vm_module = ireert.VmModule.from_buffer(
        config.vm_instance, flatbuffer_blob, warn_if_copy=False
    )
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(vm_module)
    ModuleCompiled = getattr(ctx.modules, vm_module.name)
    return ModuleCompiled, config


def load_vmfb_using_mmap(
    flatbuffer_blob_or_path, device: str, device_idx: int = None
):
    print(f"Loading module {flatbuffer_blob_or_path}...")
    if "rocm" in device:
        device = "rocm"
    with DetailLogger(timeout=2.5) as dl:
        # First get configs.
        if device_idx is not None:
            dl.log(f"Mapping device id: {device_idx}")
            device = iree_device_map(device)
            haldriver = ireert.get_driver(device)
            dl.log(f"ireert.get_driver()")

            haldevice = haldriver.create_device(
                haldriver.query_available_devices()[device_idx]["device_id"],
                allocators=shark_args.device_allocator,
            )
            dl.log(f"ireert.create_device()")
            config = ireert.Config(device=haldevice)
            dl.log(f"ireert.Config()")
        else:
            config = get_iree_runtime_config(device)
            dl.log("get_iree_runtime_config")
        if "task" in device:
            print(
                f"[DEBUG] setting iree runtime flags for cpu:\n{' '.join(get_iree_cpu_rt_args())}"
            )
            for flag in get_iree_cpu_rt_args():
                ireert.flags.parse_flags(flag)
        # Now load vmfb.
        # Two scenarios we have here :-
        #      1. We either have the vmfb already saved and therefore pass the path of it.
        #         (This would arise if we're invoking `load_module` from a SharkInference obj)
        #   OR 2. We are compiling on the fly, therefore we have the flatbuffer blob to play with.
        #         (This would arise if we're invoking `compile` from a SharkInference obj)
        temp_file_to_unlink = None
        if isinstance(flatbuffer_blob_or_path, Path):
            flatbuffer_blob_or_path = flatbuffer_blob_or_path.__str__()
        if (
            isinstance(flatbuffer_blob_or_path, str)
            and ".vmfb" in flatbuffer_blob_or_path
        ):
            vmfb_file_path = flatbuffer_blob_or_path
            mmaped_vmfb = ireert.VmModule.mmap(
                config.vm_instance, flatbuffer_blob_or_path
            )
            dl.log(f"mmap {flatbuffer_blob_or_path}")
            ctx = ireert.SystemContext(config=config)
            dl.log(f"ireert.SystemContext created")
            if "vulkan" in device:
                # Vulkan pipeline creation consumes significant amount of time.
                print(
                    "\tCompiling Vulkan shaders. This may take a few minutes."
                )
            ctx.add_vm_module(mmaped_vmfb)
            dl.log(f"module initialized")
            mmaped_vmfb = getattr(ctx.modules, mmaped_vmfb.name)
        else:
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.write(flatbuffer_blob_or_path)
                tf.flush()
                vmfb_file_path = tf.name
            temp_file_to_unlink = vmfb_file_path
            mmaped_vmfb = ireert.VmModule.mmap(instance, vmfb_file_path)
            dl.log(f"mmap temp {vmfb_file_path}")
        return mmaped_vmfb, config, temp_file_to_unlink


def get_iree_compiled_module(
    module,
    device: str,
    frontend: str = "torch",
    model_config_path: str = None,
    extra_args: list = [],
    device_idx: int = None,
    mmap: bool = False,
    debug: bool = False,
):
    """Given a module returns the compiled .vmfb and configs"""
    flatbuffer_blob = compile_module_to_flatbuffer(
        module, device, frontend, model_config_path, extra_args, debug
    )
    temp_file_to_unlink = None
    # TODO: Currently mmap=True control flow path has been switched off for mmap.
    #       Got to find a cleaner way to unlink/delete the temporary file since
    #       we're setting delete=False when creating NamedTemporaryFile. That's why
    #       I'm getting hold of the name of the temporary file in `temp_file_to_unlink`.
    if mmap:
        vmfb, config, temp_file_to_unlink = load_vmfb_using_mmap(
            flatbuffer_blob, device, device_idx
        )
    else:
        vmfb, config = get_iree_module(
            flatbuffer_blob, device, device_idx=device_idx
        )
    ret_params = {
        "vmfb": vmfb,
        "config": config,
        "temp_file_to_unlink": temp_file_to_unlink,
    }
    return ret_params


def load_flatbuffer(
    flatbuffer_path: str,
    device: str,
    device_idx: int = None,
    mmap: bool = False,
):
    temp_file_to_unlink = None
    if mmap:
        vmfb, config, temp_file_to_unlink = load_vmfb_using_mmap(
            flatbuffer_path, device, device_idx
        )
    else:
        with open(os.path.join(flatbuffer_path), "rb") as f:
            flatbuffer_blob = f.read()
        vmfb, config = get_iree_module(
            flatbuffer_blob, device, device_idx=device_idx
        )
    ret_params = {
        "vmfb": vmfb,
        "config": config,
        "temp_file_to_unlink": temp_file_to_unlink,
    }
    return ret_params


def export_iree_module_to_vmfb(
    module,
    device: str,
    directory: str,
    mlir_dialect: str = "linalg",
    model_config_path: str = None,
    module_name: str = None,
    extra_args: list = [],
    debug: bool = False,
):
    # Compiles the module given specs and saves it as .vmfb file.
    flatbuffer_blob = compile_module_to_flatbuffer(
        module, device, mlir_dialect, model_config_path, extra_args, debug
    )
    if module_name is None:
        device_name = (
            device if "://" not in device else "-".join(device.split("://"))
        )
        module_name = f"{mlir_dialect}_{device_name}"
    filename = os.path.join(directory, module_name + ".vmfb")
    with open(filename, "wb") as f:
        f.write(flatbuffer_blob)
    print(f"Saved vmfb in {filename}.")
    return filename


def export_module_to_mlir_file(module, frontend, directory: str):
    # TODO: write proper documentation.
    mlir_str = module
    if frontend in ["tensorflow", "tf", "mhlo", "stablehlo", "tflite"]:
        mlir_str = module.decode("utf-8")
    elif frontend in ["pytorch", "torch"]:
        mlir_str = module.operation.get_asm()
    filename = os.path.join(directory, "model.mlir")
    with open(filename, "w") as f:
        f.write(mlir_str)
    print(f"Saved mlir in {filename}.")
    return filename


def get_results(
    compiled_vm,
    function_name,
    input,
    config,
    frontend="torch",
    send_to_host=True,
    debug_timeout: float = 5.0,
):
    """Runs a .vmfb file given inputs and config and returns output."""
    with DetailLogger(debug_timeout) as dl:
        device_inputs = []
        for input_array in input:
            dl.log(f"Load to device: {input_array.shape}")
            device_inputs.append(
                ireert.asdevicearray(config.device, input_array)
            )
        dl.log(f"Invoke function: {function_name}")
        result = compiled_vm[function_name](*device_inputs)
        dl.log(f"Invoke complete")
        result_tensors = []
        if isinstance(result, tuple):
            if send_to_host:
                for val in result:
                    dl.log(f"Result to host: {val.shape}")
                    result_tensors.append(np.asarray(val, val.dtype))
            else:
                for val in result:
                    result_tensors.append(val)
            return result_tensors
        elif isinstance(result, dict):
            data = list(result.items())
            if send_to_host:
                res = np.array(data, dtype=object)
                return np.copy(res)
            return data
        else:
            if send_to_host and result is not None:
                dl.log("Result to host")
                return result.to_host()
            return result
        dl.log("Execution complete")


@functools.cache
def get_iree_runtime_config(device):
    device = iree_device_map(device)
    haldriver = ireert.get_driver(device)
    if device == "metal" and shark_args.device_allocator == "caching":
        print(
            "[WARNING] metal devices can not have a `caching` allocator."
            "\nUsing default allocator `None`"
        )
    haldevice = haldriver.create_device_by_uri(
        device,
        # metal devices have a failure with caching allocators atm. blcking this util it gets fixed upstream.
        allocators=shark_args.device_allocator if device != "metal" else None,
    )
    config = ireert.Config(device=haldevice)
    return config
