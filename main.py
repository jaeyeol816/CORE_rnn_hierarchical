import argparse
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_trainer, init_seed, set_color
from core_ave import COREave
from core_trm import COREtrm
from core_rnn import CORErnn
from core_hierarchical import COREhierarchical  # 새로 추가

def run_single_model(args):
    if args.model == 'ave':
        model_class = COREave
        config_file = 'props/core_ave.yaml'
    elif args.model == 'trm':
        model_class = COREtrm
        config_file = 'props/core_trm.yaml'
    elif args.model == 'rnn':
        model_class = CORErnn
        config_file = 'props/core_rnn.yaml'
    elif args.model == 'hierarchical':
        model_class = COREhierarchical
        config_file = 'props/core_hierarchical.yaml'
    else:
        raise ValueError('model can only be "ave", "trm", "rnn", or "hierarchical".')

    config = Config(
        model=model_class,
        dataset=args.dataset, 
        config_file_list=['props/overall.yaml', config_file]
    )
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading
    model = model_class(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # fit
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )
    
    # evaluate
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='trm', help='ave or trm or rnn or hierarchical')
    parser.add_argument('--dataset', type=str, default='diginetica', help='diginetica, nowplaying, retailrocket, tmall, yoochoose')
    args, _ = parser.parse_known_args()

    run_single_model(args)
