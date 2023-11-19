#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

# hijack tpu-mlir's original `run_calibration.py`

import argparse
import pymlir

from calibration.kld_calibrator import SimpleTuner, ActivationCalibrator2
from calibration.kld_calibrator import *
from calibration.data_selector import DataSelector


class SimpleTuner_hijack(SimpleTuner):

    def load_net_input(self):
        self.dq_activations = {}
        self.ref_activations = {}
        inp_ref_dict = {}
        for input in self.module.input_names:
            inp_ref_dict[input] = self.parser.get_use_count_by_op_name(input)

        assert self.ds.all_image     # sir, this way :)
        batched_inputs = self.input_num * ['']

        idx, tune_idx = 0, 0
        self.dq_activations[tune_idx] = {}
        self.ref_activations[tune_idx] = {}
        print(f'prepare data from {len(self.data_list)}')
        for data in self.data_list:
            if len(self.ref_activations) > self.args.tune_num + 1: break

            idx += 1
            inputs = data.split(',')
            inputs = [s.strip() for s in inputs]
            assert (self.input_num == len(inputs))
            for i in range(self.input_num):
                batched_inputs[i] += '{},'.format(inputs[i])
                if idx == self.batch_size:
                    # x: [1, 3, 192, 256], float32, vrng [0, 255]
                    x = self.ppa_list[i].run(batched_inputs[i][:-1])
                    if 'rgb2y':
                        x = np.transpose(x[0], [1, 2, 0])   # [H, W, C]
                        x /= 255.0                          # float32, [0.0, 1.0]
                        x = x[:, :, ::-1]                   # rgb2bgr
                        x = np.dot(x, [                     # RGB => Y
                            24.966, 128.553, 65.481
                        ]) + 16.0 
                        x = x[None, None , :, :]
                    # x: [1, 1, 192, 256], float32, vrng [0, 255]
                    name = self.ppa_list[i].input_name
                    self.dq_activations[tune_idx][name] = [x, inp_ref_dict[name]]
                    self.ref_activations[tune_idx][name] = [x, inp_ref_dict[name]]
            if idx == self.batch_size:
                idx = 0
                batched_inputs = self.input_num * ['']
            else:
                continue

            tune_idx += 1
            self.dq_activations[tune_idx] = {}
            self.ref_activations[tune_idx] = {}

        if len(self.ref_activations[tune_idx]) == 0:
            # print(f'last tune data (tune_idx={tune_idx}) not valid, droped')
            self.ref_activations.pop(tune_idx)
        self.args.tune_num = min(self.args.tune_num, len(self.ref_activations))
        # print(f"tune_num = {self.args.tune_num}, ref = {len(self.ref_activations)}")
        # print(f"real tune_num = {self.args.tune_num}")
        assert self.args.tune_num > 0


class ActivationCalibrator2_hijack(ActivationCalibrator2):

    def load_net_input(self):
        self.dq_activations = {}
        self.ref_activations = {}
        inp_ref_dict = {}
        for input in self.module.input_names:
            inp_ref_dict[input] = self.parser.get_use_count_by_op_name(input)

        assert self.ds.all_image     # sir, this way :)
        batched_inputs = self.input_num * ['']

        idx, tune_idx = 0, 0
        self.dq_activations[tune_idx] = {}
        self.ref_activations[tune_idx] = {}
        for data in self.data_list:
            idx += 1
            inputs = data.split(',')    # img paths
            inputs = [s.strip() for s in inputs]
            assert (self.input_num == len(inputs))
            for i in range(self.input_num):     # self.input_num == 1
                # str: img paths
                batched_inputs[i] += '{},'.format(inputs[i])
                if idx == self.batch_size:      # self.batch_size == 1
                    # x: [1, 3, 192, 256], float32, vrng [0, 255]
                    x = self.ppa_list[i].run(batched_inputs[i][:-1])
                    if 'rgb2y':
                        x = np.transpose(x[0], [1, 2, 0])   # [H, W, C]
                        x /= 255.0                          # float32, [0.0, 1.0]
                        x = x[:, :, ::-1]                   # rgb2bgr
                        x = np.dot(x, [                     # RGB => Y
                            24.966, 128.553, 65.481
                        ]) + 16.0 
                        x = x[None, None , :, :]
                    # x: [1, 1, 192, 256], float32, vrng [0, 255]
                    name = self.ppa_list[i].input_name
                    self.dq_activations[tune_idx][name] = [x, inp_ref_dict[name]]
                    self.ref_activations[tune_idx][name] = [x, inp_ref_dict[name]]
            if idx == self.batch_size:
                idx = 0
                batched_inputs = self.input_num * ['']
            else:
                continue

            tune_idx += 1
            self.dq_activations[tune_idx] = {}
            self.ref_activations[tune_idx] = {}

        if len(self.ref_activations[tune_idx]) == 0:
            print(f'last input data (idx={tune_idx}) not valid, droped')
            self.ref_activations.pop(tune_idx)
        
        self.args.input_num = min(self.args.input_num, len(self.ref_activations))
        print(f"input_num = {self.args.input_num}, ref = {len(self.ref_activations)}")
        print(f"real input_num = {self.args.input_num}")
        assert self.args.input_num > 0

    def run(self):
        layer_name_list = []
        thresholds_map_list = []
        op_layers = self.parser.get_op_name_list()
        if 'input_calibration_table' in self.debug_cmd:
            assert self.args.tune_num > 0
            input_calibration_table = self.debug_cmd['input_calibration_table']
            if input_calibration_table != '' and os.path.exists(input_calibration_table):
                os.system('cp -f {name} {name}.1'.format(name=input_calibration_table))
                threshold_table = CalibrationTable(input_calibration_table)
                for op_name in op_layers:
                    thresholds_map_list.append(threshold_table.thresholds_map[op_name][0])
            else:
                print('input_calibration_table error')
                exit(1)
        else:
            thresholds_map, thresholds_map_absmax, thresholds_map_scale, thresholds_map_zp, thresholds_map4, thresholds_map_absmax4, thresholds_map_scale4, thresholds_map_zp4 = self.activation_collect_and_calc_th()
            self._clean_resource()
            # step 3: dump threshold table of default histogram bins
            cali_table = self.args.calibration_table
            if self.args.tune_num > 0:
                cali_table += ".1"
            with open(cali_table, 'w') as f:
                f.write("# genetated time: {}\n".format(datetime.datetime.now()))
                f.write("# histogram number: {}\n".format(self.histogram_bin_num))
                f.write("# sample number: {}\n###\n".format(self.num_samples))
                f.write("# op_name    threshold    min    max\n")
                for i, op_name in enumerate(op_layers):
                    if 'int4' in self.debug_cmd:
                        if 'use_torch_observer_for_cali' in self.debug_cmd:
                            qmin, qmax = -128, 127
                            scale = thresholds_map_scale[op_name]
                            zp = thresholds_map_zp[op_name]
                            threshold = float(scale * max(-(qmin-zp), qmax-zp))
                            min_value = -threshold*128.0/127.0
                            max_value = threshold
                        else:
                            threshold = thresholds_map[op_name]
                            min_value, max_value = -threshold*128.0/127.0, threshold
                        thresholds_map_list.append(threshold)
                        f.write("{} {:.7f} {:.7f} {:.7f}\n".format(op_name, threshold, min_value,
                                                               max_value))
                    else:
                        if 'use_torch_observer_for_cali' in self.debug_cmd:
                            qmin, qmax = -128, 127
                            scale = thresholds_map_scale[op_name]
                            zp = thresholds_map_zp[op_name]
                            threshold = float(scale * max(-(qmin-zp), qmax-zp))
                            min_value = float(scale * (qmin - zp))
                            max_value = float(scale * (qmax - zp))
                        else:
                            if op_name in thresholds_map:
                                threshold = thresholds_map[op_name]
                            else:
                                threshold = 1.0
                            if op_name in self.activations_statistics:
                                min_value, max_value, _ = self.activations_statistics[op_name]
                            else:
                                min_value, max_value = -1,1
                        thresholds_map_list.append(threshold)
                        f.write("{} {:.7f} {:.7f} {:.7f}\n".format(op_name, threshold, min_value, max_value))
                if 'int4' in self.debug_cmd and ('use_torch_observer_for_cali' in self.debug_cmd or 'use_percentile9999' in self.debug_cmd or 'use_max' in self.debug_cmd):
                    f.write("\n")
                    f.write("#int4_th\n")
                    for i, op_name in enumerate(op_layers):
                        if 'use_torch_observer_for_cali' in self.debug_cmd:
                            qmin, qmax = -8, 7
                            scale = thresholds_map_scale4[op_name]
                            zp = thresholds_map_zp4[op_name]
                            threshold = float(scale * max(-(qmin-zp), qmax-zp))
                            min_value = -threshold*128.0/127.0
                            max_value = threshold
                        else:
                            threshold = thresholds_map4[op_name]
                            min_value, max_value = -threshold*128.0/127.0, threshold
                        f.write("{} {:.7f} {:.7f} {:.7f}\n".format(op_name, threshold, min_value, max_value))

            # if 'use_torch_observer_for_cali' in self.debug_cmd:
            #     exit(0)
        if self.args.tune_num <= 0 or 'int4' in self.debug_cmd:
            return

        # setp 4: tune to get better threshold of each layers.
        self.tunner = SimpleTuner_hijack(self.args, self.tune_ds, self.ppa_list, thresholds_map_absmax)
        thresholds = self.tunner.run()

        # step 5: dump threshold table after tuning
        tuned_threshold_list = []
        with open(self.args.calibration_table, 'w') as f:
            f.write("# genetated time: {}\n".format(datetime.datetime.now()))
            f.write("# histogram number: {}\n".format(self.histogram_bin_num))
            f.write("# sample number: {}\n".format(self.num_samples))
            f.write("# tune number: {}\n###\n".format(self.args.tune_num))
            f.write("# op_name    threshold    min    max\n")
            for i, op_name in enumerate(op_layers):
                threshold = thresholds[op_name]
                layer_name_list.append('{}_{}'.format(i, op_name))
                tuned_threshold_list.append(threshold)
                if 'input_calibration_table' in self.debug_cmd:
                    min_value = threshold_table.thresholds_map[op_name][1]
                    max_value = threshold_table.thresholds_map[op_name][2]
                else:
                    if 'use_torch_observer_for_cali' in self.debug_cmd:
                        qmin, qmax = -128, 127
                        scale = thresholds_map_scale[op_name]
                        zp = thresholds_map_zp[op_name]
                        min_value = float(scale * (qmin - zp))
                        max_value = float(scale * (qmax - zp))
                    else:
                        min_value, max_value, _ = self.activations_statistics[op_name]
                f.write("{} {:.7f} {:.7f} {:.7f}\n".format(op_name, threshold, min_value, max_value))
        os.remove(cali_table)
        if 'print_debug_info' in self.debug_cmd:
            th_before_tuned = np.array(thresholds_map_list)
            th_after_tuned = np.array(tuned_threshold_list)
            file_prefix = './{}_{}pics_{}_times_tuned_th_statistic'.format(self.args.mlir_file.split('.')[0], self.tunner.args.tune_num, self.tunner.tune_steps)
            save_tensor_diff_subplot(th_before_tuned, th_after_tuned, layer_name_list, 'before_tuned', 'after_tuned', file_prefix)


if __name__ == '__main__':
    print("SOPHGO Toolchain {}".format(pymlir.module().version))
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument('mlir_file', metavar='mlir_file', help='mlir file')
    parser.add_argument('--dataset', type=str, help='dataset for calibration')
    parser.add_argument('--data_list', type=str, help='Input list file contain all input')
    parser.add_argument('--input_num', type=int, default=0, help='num of images for calibration')
    parser.add_argument('--tune_list', type=str, default='', help='Tune list file contain all input for tune')
    parser.add_argument('--tune_num', type=int, default=5, help='num of images for tune')
    parser.add_argument('--histogram_bin_num', type=int, default=2048, help='Specify histogram bin numer for kld calculate')
    parser.add_argument('-o', '--calibration_table', type=str, help='output threshold table')
    parser.add_argument('--debug_cmd', type=str, default='', help='debug cmd')
    # yapf: enable
    args = parser.parse_args()
    dump_list = True if 'dump_list' in args.debug_cmd else False
    selector = DataSelector(args.dataset, args.input_num, args.data_list)
    tune_ds = None
    if args.tune_list:
        tune_ds = DataSelector(None, args.tune_num, args.tune_list)
        args.tune_num = len(tune_ds.data_list)
    if dump_list:
        selector.dump("./selected_image_list.txt")
        if tune_ds is not None:
            tune_ds.dump("./selected_tune_image_list.txt")
    # calibration
    calibrator = ActivationCalibrator2_hijack(args, selector, tune_ds)
    calibrator.run()
