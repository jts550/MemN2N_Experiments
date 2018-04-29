import argparse
import trainer
import numpy as np
import pdb
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--dataset_dir", type=str, default="bAbI/tasks_1-20_v1-2/en/")
    parser.add_argument("--task", type=int, default=3)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--decay_interval", type=int, default=25)
    parser.add_argument("--decay_ratio", type=float, default=0.5)
    parser.add_argument("--max_clip", type=float, default=40.0)

    return parser.parse_args()


def main(config):
    accs = []
    models = []
    accs_ensemble = []
    correct_ensemble = []
    pred_prob_ensemble = []
    answer_ensemble = []
    for _ in range(100):
        t = trainer.Trainer(config)
        acc = t.fit()
        accs.append(acc)
        print(np.array(accs).mean())
        models.append(t)
        accs_individual, correct, pred_prob, answer = t.evaluate('test', ensemble = True)
        accs_ensemble.append(accs_individual)
        correct_ensemble.append(correct)
        pred_prob_ensemble.append(pred_prob)
        answer_ensemble.append(answer)
        init_prob = np.ones_like(pred_prob_ensemble[0])
        for p in pred_prob_ensemble:
            init_prob = init_prob * p
        guess = init_prob.argmax(1)
        ensemble_total_accuracy = (guess == answer_ensemble[0]).mean()
        print('ensemble_accuracy_so_far', ensemble_total_accuracy)
        pred_prob_ensemble[0].argmax(1) == answer_ensemble[0]
    #pdb.set_trace()
    npaccs = np.array(accs)
    print(npaccs.mean(),npaccs.std(),npaccs.min())

if __name__ == "__main__":
    config = parse_config()
    main(config)
