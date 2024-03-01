# from metrics import mi_metric, te_metric, nmi_metric
# from utils import read_grey_tensor

# # ir_tensor = [read_grey_tensor(dataset='TNO',category='ir',name=str(i+1)+'.bmp',requires_grad=False) for i in range(10)]
# # vis_tensor = [read_grey_tensor(dataset='TNO',category='vis',name=str(i+1)+'.bmp',requires_grad=False) for i in range(10)]
# # fuse_tensor = [read_grey_tensor(dataset='TNO',category='fuse',name=str(i+1)+'.bmp',requires_grad=False) for i in range(10)]

# mi_scores = []
# te_scores = []
# nmi_scores = []
# temp = []
# temp2 = []

# def normalize_scores(scores):
#     min_score = min(scores)
#     max_score = max(scores)
#     return [(score - min_score) / (max_score - min_score) for score in scores]

# for method in ['ADF','CBF','CNN','FPDE','GFCE','GTF','HMSD_GF','IFEVIP','LatLRR','MSVD','TIF','U2Fusion','VSMWLS']:
#     ir = read_grey_tensor(dataset='TNO',category='ir',name='9.bmp',requires_grad=False)
#     vis = read_grey_tensor(dataset='TNO',category='vis',name='9.bmp',requires_grad=False)
#     fuse = read_grey_tensor(dataset='TNO',category='fuse',model=method,name='9.bmp',requires_grad=False)
#     mi_scores.append(mi_metric(vis,ir,fuse))
#     te_scores.append(te_metric(vis,ir,fuse))
#     nmi_scores.append(nmi_metric(vis,ir,fuse))
#     # temp.append(ag_metric(vis,ir,fuse))
#     # temp2.append(ag_metric(vis,ir,fuse)+1)

# mi_scores = normalize_scores(mi_scores)
# te_scores = normalize_scores(te_scores)
# nmi_scores = normalize_scores(nmi_scores)

# # Combine and sort scores based on MI
# combined_scores = list(zip(mi_scores_normalized, te_scores_normalized, nmi_scores_normalized))
# combined_scores.sort(key=lambda x: x[0])

# # Unzip sorted scores
# mi_scores_normalized, te_scores_normalized, nmi_scores_normalized = zip(*combined_scores)

# print(mi_scores)
# print(te_scores)
# print(nmi_scores)
# plt.plot(mi_scores)
# plt.plot(te_scores)
# plt.show()
