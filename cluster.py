from dataloader import *
from util import *
from model import *
from init_parameter import *

def labels_to_index(labels):
    label_to_index = {k: i for i, k in enumerate(labels)}
    index_to_label = {i: k for i, k in enumerate(labels)}
    return label_to_index, index_to_label

def unknown_labels_index(known_labels, label_to_index, index_to_label):
    unknow_to_index = {k: v for k, v in label_to_index.items() if k not in known_labels}
    index_to_unknown = {k: v for k, v in index_to_label.items() if v not in known_labels}
    return unknow_to_index, index_to_unknown

def get_unknown_labels(known_labels, labels):
    return [label for label in labels if label not in known_labels]

def map(ind, pseudo_labels):
    m = {i[0]: i[1] for i in ind}
    y_pred = np.array([m[idx] for idx in pseudo_labels])
    return y_pred


def get_features_labels(dataloader, model, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    total_features = torch.empty((0,args.feat_dim)).to(device)
    total_labels = torch.empty(0,dtype=torch.long).to(device)

    for batch in tqdm(dataloader, desc="Extracting representation"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, label_ids = batch
        with torch.no_grad():
            feature = model(input_ids, input_mask, feature_ext = True)

        total_features = torch.cat((total_features, feature))
        total_labels = torch.cat((total_labels, label_ids))

    return total_features, total_labels

def save_result(true_labels, pseudo_labels, examples, labels, known_labels):
    label_to_index, index_to_label = labels_to_index(labels)

    ind, w = hungray_aligment(true_labels, pseudo_labels)
    pred_labels = map(ind, pseudo_labels)

    # Save confussion matrix
    # conf_mtx = confusion_matrix(pred_labels, true_labels)
    # index = [l + ' (pred)' for l in labels]
    df = pd.DataFrame(w, columns=labels, index=[i for i in range(np.max(pseudo_labels) + 1)])
    df.to_csv('conf_mtx.csv', index_label='Cluster id')

    # Save full intent cluster
    cluster_id_to_label_id = np.argmax(w, axis=1)
    cluster_id_to_labels = {i : index_to_label[label_id] for i, label_id in enumerate(cluster_id_to_label_id)}

    # pred_labels = np.array([cluster_id_to_label_id[label] for label in pseudo_labels])

    true_false = (pred_labels == true_labels)
    print(f'Accuracy: {true_false.sum() / len(true_labels)}')

    with open('full_intent.csv', 'w') as f:
        f.write('User_say;True_intent;Cluster_id;Pred_intent;True/False\n')
        for i, ex in enumerate(examples):
            f.write(ex.text_a + ';' + ex.label + ';' + str(pseudo_labels[i]) + ';' + index_to_label[pred_labels[i]] + ';' + str(true_false[i]) + '\n')

    # Save new intent cluster
    unknown_labels = get_unknown_labels(known_labels, labels)
    unknown_examples_pos = [i for i, ex in enumerate(examples) if ex.label in unknown_labels]

    unknow_to_index, index_to_unknown = unknown_labels_index(known_labels, label_to_index, index_to_label)
    with open('new_intent_pred.csv', 'w') as f:
        f.write('User_say;True_intent;Cluter_id;Pred_intent;True/False\n')
        for pos in unknown_examples_pos:
            f.write(examples[pos].text_a + ';' + examples[pos].label + ';' + str(pseudo_labels[pos]) + ';' + index_to_label[pred_labels[pos]] +  ';' + str(true_false[pos]) + '\n')


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = init_model()
    args = parser.parse_args()
    data_dir = os.path.join(args.data_dir, args.dataset)
    data = Data(args)
    data_full = data.get_type(args, 'full')

    labels = data.processor.get_labels(data_dir)
    known_labels = data.processor.get_known_labels(data_dir)
    full_examples = data.processor.get_examples(data_dir, 'full')

    unknown_labels = get_unknown_labels(known_labels, labels)
    unknown_examples_pos = [i for i, ex in enumerate(full_examples) if ex.label  in unknown_labels]

    model = BertForModel.from_pretrained(args.bert_model, cache_dir = "", n_labels=data.num_labels)
    model.load_state_dict(torch.load(os.path.join(args.final_model_dir, WEIGHTS_NAME), map_location=torch.device('cpu')))
    model.to(device)

    feats, true_labels = get_features_labels(data_full, model, args)
    feats = feats.cpu().numpy()
    km = Kmeans(feats.shape[1], args.K, niter=100, verbose=True)
    km.train(feats)
    D, I = km.index.search(feats, 1)
    pseudo_labels = I.flatten()

    true_labels = true_labels.cpu().numpy().flatten()
    save_result(true_labels, pseudo_labels, full_examples, labels, known_labels)

    # label_to_index, index_to_label = labels_to_index(labels)
    # print(index_to_label)

    # _, i_to_l = get_labels_index(data.processor.get_labels(data_dir))
    # print([i_to_l[y] for y in y_pred])





