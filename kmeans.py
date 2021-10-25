from dataloader import *
from util import *
from model import *
from init_parameter import *
from transformers import AutoModel
from sklearn.cluster import SpectralClustering, DBSCAN


def get_features_labels(dataloader, model, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    total_features = torch.empty((0, 768)).to(device)
    total_labels = torch.empty(0,dtype=torch.long).to(device)

    for batch in tqdm(dataloader, desc="Extracting representation"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, label_ids = batch
        with torch.no_grad():
            feature = model(input_ids, attention_mask=input_mask)[0].mean(dim=1)

        total_features = torch.cat((total_features, feature))
        total_labels = torch.cat((total_labels, label_ids))
    return total_features, total_labels


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = init_model()
    args = parser.parse_args()
    data_dir = os.path.join(args.data_dir, args.dataset)
    data = Data(args)
    data_full = data.get_type(args, 'full')
    phobert = AutoModel.from_pretrained(args.bert_model).to(device)
    feats, labels = get_features_labels(data_full, phobert, args)
    feats = feats.cpu().numpy()
    feats = feats.astype(np.float32)
    km = Kmeans(feats.shape[1], data.num_labels, nredo=10, niter=args.num_kmean_iter, verbose=True, gpu=True)
    km.train(feats)

    _, I = km.index.search(feats, 1)
    y_pred = I.flatten()
    # clustering = DBSCAN(eps=0.5,min_samples=20).fit(feats)
    y_true = labels.cpu().numpy()
    # y_pred = km.index.search(feats, 1)
    results = clustering_score(y_true, y_pred)
    print('score',results)
