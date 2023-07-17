#!/usr/bin/python
import re
import argparse
import numpy as np
import logging
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

from tritonclient.utils import np_to_triton_dtype

FLAGS = None


def response_refine(resp):
    resp = re.sub(r'\s*#+\s+Instruction\s*:.*', '', resp)
    resp = re.sub(r'\s*#+\s+Response\s*:.*', '', resp)
    resp = re.sub(r'\n+', '\n', resp).strip()
    return resp


def prepare_tensor(name, input, protocol):
    client_util = httpclient if protocol == "http" else grpcclient
    t = client_util.InferInput(
        name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def create_inference_server_client(protocol, url, concurrency, verbose):
    client_util = httpclient if protocol == "http" else grpcclient
    if protocol == "http":
        return client_util.InferenceServerClient(url,
                                                 concurrency=concurrency,
                                                 verbose=verbose)
    elif protocol == "grpc":
        return client_util.InferenceServerClient(url, verbose=verbose)


def append_start_and_end_ids(inputs, batch_size, start_id=None, end_id=None):
    if start_id is not None:
        start_ids = start_id * np.ones([batch_size, 1]).astype(np.uint32)
        inputs.append(prepare_tensor("start_id", start_ids, FLAGS.protocol))
    if end_id is not None:
        end_ids = end_id * np.ones([batch_size, 1]).astype(np.uint32)
        inputs.append(prepare_tensor("end_id", end_ids, FLAGS.protocol))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        help='Inference server URL.')
    parser.add_argument('-beam',
                        '--beam_width',
                        type=int,
                        default=1,
                        help='beam width.')
    parser.add_argument('-topk',
                        '--topk',
                        type=int,
                        default=1,
                        required=False,
                        help='topk for sampling')
    parser.add_argument('-topp',
                        '--topp',
                        type=float,
                        default=0.0,
                        required=False,
                        help='topp for sampling')
    parser.add_argument('-i',
                        '--protocol',
                        type=str,
                        required=False,
                        default='http',
                        help='Protocol ("http"/"grpc") used to '
                             'communicate with inference service. Default is "http".')
    parser.add_argument('--return_log_probs',
                        action="store_true",
                        default=False,
                        required=False,
                        help='return the cumulative log probs and output log probs or not')
    parser.add_argument('--output-len',
                        type=int,
                        default=24,
                        required=False,
                        help='the number of tokens we hope model generating')
    FLAGS = parser.parse_args()

    LOGGER = logging.getLogger(f"{__file__} {__name__}")
    log_format = "%(asctime)s %(name)s:%(lineno)d [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG if FLAGS.verbose else logging.INFO,
                        format=log_format)

    if (FLAGS.protocol != "http") and (FLAGS.protocol != "grpc"):
        LOGGER.error(f'unexpected protocol "{FLAGS.protocol}", expects "http" or "grpc"')
        exit(1)

    if FLAGS.url is None:
        FLAGS.url = "localhost:8000" if FLAGS.protocol == "http" else "localhost:8001"

    # Run async requests to make sure backend handles request batches
    # correctly. We use just HTTP for this since we are not testing the
    # protocol anyway.

    ######################
    LOGGER.info(" Preprocessing ".center(80, '='))
    model_name = "preprocessing"
    with create_inference_server_client(FLAGS.protocol,
                                        FLAGS.url,
                                        concurrency=1,
                                        verbose=FLAGS.verbose) as client:
        input0 = [
            ["Blackhawks\n The 2015 Hilltoppers"],
            ["Data sources you can use to make a decision:"],
            ["\n if(angle = 0) { if(angle"],
            ["GMs typically get 78% female enrollment, but the "],
            ["Previous Chapter | Index | Next Chapter"],
            ["Michael, an American Jew, called Jews"],
            ["Blackhawks\n The 2015 Hilltoppers"],
            ["Data sources you can use to make a comparison:"]
        ]
        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(np.uint32) * FLAGS.output_len
        bad_words_list = np.array([
            [""],
            [""],
            [""],
            [""],
            [""],
            [""],
            [""],
            [""]], dtype=object)
        stop_words_list = np.array([
            [""],
            [""],
            [""],
            [""],
            [""],
            [""],
            [""],
            [""]], dtype=object)
        inputs = [
            prepare_tensor("QUERY", input0_data, FLAGS.protocol),
            prepare_tensor("BAD_WORDS_DICT", bad_words_list, FLAGS.protocol),
            prepare_tensor("STOP_WORDS_DICT", stop_words_list, FLAGS.protocol),
            prepare_tensor("REQUEST_OUTPUT_LEN", output0_len, FLAGS.protocol),
        ]

        try:
            result = client.infer(model_name, inputs)
            output0 = result.as_numpy("INPUT_ID")
            output1 = result.as_numpy("REQUEST_INPUT_LEN")
            output2 = result.as_numpy("REQUEST_OUTPUT_LEN")
            output3 = result.as_numpy("BAD_WORDS_IDS")
            output4 = result.as_numpy("STOP_WORDS_IDS")
            LOGGER.info("============After preprocessing============")
            LOGGER.info(f"INPUT_ID: \n {output0}")
            LOGGER.info(f"REQUEST_INPUT_LEN: \n {output1}")
            LOGGER.info(f"REQUEST_OUTPUT_LEN: \n {output2}")
            LOGGER.info(f"BAD_WORDS_IDS: \n {output3}")
            LOGGER.info(f"STOP_WORDS_IDS: \n {output4}")
            LOGGER.info("===========================================\n\n\n")
        except Exception as e:
            LOGGER.error(e)

    ######################
    LOGGER.info(" Faster Transformer ".center(80, '='))
    model_name = "fastertransformer"
    with create_inference_server_client(FLAGS.protocol,
                                        FLAGS.url,
                                        concurrency=1,
                                        verbose=FLAGS.verbose) as client:
        runtime_top_k = (FLAGS.topk * np.ones([output0.shape[0], 1])).astype(np.uint32)
        runtime_top_p = FLAGS.topp * np.ones([output0.shape[0], 1]).astype(np.float32)
        beam_search_diversity_rate = 0.0 * np.ones([output0.shape[0], 1]).astype(np.float32)
        temperature = 1.0 * np.ones([output0.shape[0], 1]).astype(np.float32)
        len_penalty = 1.0 * np.ones([output0.shape[0], 1]).astype(np.float32)
        repetition_penalty = 1.0 * np.ones([output0.shape[0], 1]).astype(np.float32)
        random_seed = 0 * np.ones([output0.shape[0], 1]).astype(np.uint64)
        is_return_log_probs = FLAGS.return_log_probs * np.ones([output0.shape[0], 1]).astype(bool)
        beam_width = (FLAGS.beam_width * np.ones([output0.shape[0], 1])).astype(np.uint32)
        prompt_learning_task_name_ids = 0 * np.ones([output0.shape[0], 1]).astype(np.uint32)
        inputs = [
            prepare_tensor("input_ids", output0, FLAGS.protocol),
            prepare_tensor("input_lengths", output1, FLAGS.protocol),
            prepare_tensor("request_output_len", output2, FLAGS.protocol),
            prepare_tensor("runtime_top_k", runtime_top_k, FLAGS.protocol),
            prepare_tensor("runtime_top_p", runtime_top_p, FLAGS.protocol),
            prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate, FLAGS.protocol),
            prepare_tensor("temperature", temperature, FLAGS.protocol),
            prepare_tensor("len_penalty", len_penalty, FLAGS.protocol),
            prepare_tensor("repetition_penalty", repetition_penalty, FLAGS.protocol),
            prepare_tensor("random_seed", random_seed, FLAGS.protocol),
            prepare_tensor("is_return_log_probs", is_return_log_probs, FLAGS.protocol),
            prepare_tensor("beam_width", beam_width, FLAGS.protocol),
            prepare_tensor("bad_words_list", output3, FLAGS.protocol),
            prepare_tensor("stop_words_list", output4, FLAGS.protocol),
        ]

        # factual-nucleus arguments
        # top_p_decay = 0.9 * np.ones([output0.shape[0], 1]).astype(np.float32)
        # top_p_min = 0.5 * np.ones([output0.shape[0], 1]).astype(np.float32)
        # top_p_reset_ids = 13 * np.ones([output0.shape[0], 1]).astype(np.uint32)
        # inputs.append(prepare_tensor("top_p_decay", top_p_decay, FLAGS.protocol))
        # inputs.append(prepare_tensor("top_p_min", top_p_min, FLAGS.protocol))
        # inputs.append(prepare_tensor("top_p_reset_ids", top_p_reset_ids, FLAGS.protocol))

        try:
            result = client.infer(model_name, inputs)
            output0 = result.as_numpy("output_ids")
            output1 = result.as_numpy("sequence_length")
            LOGGER.info("============After fastertransformer============")
            LOGGER.info(f"output_ids: \n{output0}")
            LOGGER.info(f"sequence_length: \n{output1}")
            if FLAGS.return_log_probs:
                output3 = result.as_numpy("cum_log_probs")
                output4 = result.as_numpy("output_log_probs")
                LOGGER.info(f"cum_log_probs: \n{output3}")
                LOGGER.info(f"output_log_probs: \n{output4}")
            LOGGER.info("===========================================\n\n\n")
        except Exception as e:
            LOGGER.error(e)

    ######################
    LOGGER.info(" postprocessing ".center(80, '='))
    model_name = "postprocessing"
    with create_inference_server_client(FLAGS.protocol,
                                        FLAGS.url,
                                        concurrency=1,
                                        verbose=FLAGS.verbose) as client:
        LOGGER.info(f"TOKENS_BATCH: {output0}")
        LOGGER.info(f"sequence_length: {output1}")
        inputs = [
            prepare_tensor("TOKENS_BATCH", output0, FLAGS.protocol),
            prepare_tensor("sequence_length", output1, FLAGS.protocol),
        ]
        inputs[0].set_data_from_numpy(output0)

        try:
            result = client.infer(model_name, inputs)
            output0 = (result.as_numpy("OUTPUT"))
            LOGGER.info("============After postprocessing============")
            LOGGER.info(f"output: \n{output0}")
            LOGGER.info("===========================================\n\n\n")
        except Exception as e:
            LOGGER.error(e)
