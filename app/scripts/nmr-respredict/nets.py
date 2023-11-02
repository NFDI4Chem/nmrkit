import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import util


class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth, init_std=1e-6, output_dim=None):
        # print("Creating resnet with input_dim={} hidden_dim={} depth={}".format(input_dim, hidden_dim, depth))
        print("depth=", depth)
        assert depth >= 0
        super(ResNet, self).__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        if output_dim is None:
            output_dim = hidden_dim

        self.linear_out = nn.Linear(hidden_dim, output_dim)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, init_std) for i in range(depth)]
        )

    def forward(self, input):
        input = input.view(input.size(0), -1)
        x = self.linear_in(input)
        for res_block in self.res_blocks:
            x = res_block(x)
        return self.linear_out(x)


class MaskedBatchNorm1d(nn.Module):
    def __init__(self, feature_n):
        """
        Batchnorm1d that skips some rows in the batch
        """

        super(MaskedBatchNorm1d, self).__init__()
        self.feature_n = feature_n
        self.bn = nn.BatchNorm1d(feature_n)

    def forward(self, x, mask):
        assert x.shape[0] == mask.shape[0]
        assert mask.dim() == 1

        bin_mask = mask > 0
        y_i = self.bn(x[bin_mask])
        y = torch.zeros(x.shape, device=x.device)
        y[bin_mask] = y_i
        return y


class MaskedLayerNorm1d(nn.Module):
    def __init__(self, feature_n):
        """
        LayerNorm that skips some rows in the batch
        """

        super(MaskedLayerNorm1d, self).__init__()
        self.feature_n = feature_n
        self.bn = nn.LayerNorm(feature_n)

    def forward(self, x, mask):
        assert x.shape[0] == mask.shape[0]
        assert mask.dim() == 1

        bin_mask = mask > 0
        y_i = self.bn(x[bin_mask])
        y = torch.zeros(x.shape, device=x.device)
        y[bin_mask] = y_i
        return y


class ResidualBlock(nn.Module):
    def __init__(self, dim, noise=1e-6):
        super(ResidualBlock, self).__init__()
        self.noise = noise
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim, bias=False)
        self.l1.bias.data.uniform_(-self.noise, self.noise)
        self.l1.weight.data.uniform_(-self.noise, self.noise)  # ?!
        self.l2.weight.data.uniform_(-self.noise, self.noise)

    def forward(self, x):
        return x + self.l2(F.relu(self.l1(x)))


class SumLayers(nn.Module):
    """
    Fully-connected layers that sum elements in a set
    """

    def __init__(self, input_D, input_max, filter_n, layer_count):
        super(SumLayers, self).__init__()
        self.fc1 = nn.Linear(input_D, filter_n)
        self.relu1 = nn.ReLU()
        self.fc_blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(filter_n, filter_n), nn.ReLU())
                for _ in range(layer_count - 1)
            ]
        )

    def forward(self, X, present):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional

        xt = X  # .transpose(1, 2)
        x = self.fc1(xt)
        x = self.relu1(x)
        for fcre in self.fc_blocks:
            x = fcre(x)

        x = present.unsqueeze(-1) * x

        return x.sum(1)


class ClusterCountingNetwork(nn.Module):
    """
    A network to count the number of points in each
    cluster. Very simple, mostly for pedagogy
    """

    def __init__(
        self,
        input_D,
        input_max,
        sum_fc_filternum,
        sum_fc_layercount,
        post_sum_fc,
        post_sum_layercount,
        output_dim,
    ):
        super(ClusterCountingNetwork, self).__init__()

        self.sum_layer = SumLayers(
            input_D, input_max, sum_fc_filternum, sum_fc_layercount
        )
        self.post_sum = nn.Sequential(
            ResNet(
                sum_fc_filternum,
                post_sum_fc,
                post_sum_layercount,
            ),
            nn.Linear(post_sum_fc, output_dim),
            nn.ReLU(),
        )

    def forward(self, X, present):
        sum_vals = self.sum_layer(X, present)
        return self.post_sum(sum_vals)


class ResNetRegression(nn.Module):
    def __init__(self, D, block_sizes, INT_D, FINAL_D, use_batch_norm=False, OUT_DIM=1):
        super(ResNetRegression, self).__init__()

        layers = [nn.Linear(D, INT_D)]

        for block_size in block_sizes:
            layers.append(ResNet(INT_D, INT_D, block_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(INT_D))
        layers.append(nn.Linear(INT_D, FINAL_D))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(FINAL_D, OUT_DIM))

        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)


class ResNetRegressionMaskedBN(nn.Module):
    def __init__(
        self, D, block_sizes, INT_D, FINAL_D, OUT_DIM=1, norm="batch", dropout=0.0
    ):
        super(ResNetRegressionMaskedBN, self).__init__()

        layers = [nn.Linear(D, INT_D)]
        usemask = [False]
        for block_size in block_sizes:
            layers.append(ResNet(INT_D, INT_D, block_size))
            usemask.append(False)

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
                usemask.append(False)
            if norm == "layer":
                layers.append(MaskedLayerNorm1d(INT_D))
                usemask.append(True)
            elif norm == "batch":
                layers.append(MaskedBatchNorm1d(INT_D))
                usemask.append(True)
        layers.append(nn.Linear(INT_D, OUT_DIM))
        usemask.append(False)

        self.layers = nn.ModuleList(layers)
        self.usemask = usemask

    def forward(self, x, mask):
        for l, use_mask in zip(self.layers, self.usemask):
            if use_mask:
                x = l(x, mask)
            else:
                x = l(x)
        return x


class PyTorchResNet(nn.Module):
    """
    This is a modification of the default pytorch resnet to allow
    for different input sizes, numbers of channels, kernel sizes,
    and number of block layers and classes
    """

    def __init__(
        self,
        block,
        layers,
        input_img_size=64,
        num_channels=3,
        num_classes=1,
        first_kern_size=7,
        final_avg_pool_size=7,
        inplanes=64,
    ):
        self.inplanes = inplanes
        super(PyTorchResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            num_channels,
            self.inplanes,
            kernel_size=first_kern_size,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block_layers = []
        for i, l in enumerate(layers):
            stride = 1 if i == 0 else 2
            layer = self._make_layer(block, 64 * 2**i, l, stride=stride)
            self.block_layers.append(layer)

        self.block_layers_seq = nn.Sequential(*self.block_layers)

        last_image_size = input_img_size // (2 ** (len(layers) + 1))
        post_pool_size = last_image_size - final_avg_pool_size + 1
        self.avgpool = nn.AvgPool2d(final_avg_pool_size, stride=1, padding=0)
        expected_final_planes = 32 * 2 ** len(layers)

        self.fc = nn.Linear(
            expected_final_planes * block.expansion * post_pool_size**2, num_classes
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.block_layers_seq(x)
        # for l in self.block_layers:

        #     x = l(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class SimpleGraphModel(nn.Module):
    """
    Simple graph convolution model that outputs dense features post-relu

    Add final layer for regression or classification

    """

    def __init__(
        self,
        MAX_N,
        input_feature_n,
        output_features_n,
        noise=1e-5,
        single_out_row=True,
        batch_norm=False,
        input_batch_norm=False,
    ):
        super(SimpleGraphModel, self).__init__()
        self.MAX_N = MAX_N
        self.input_feature_n = input_feature_n
        self.output_features_n = output_features_n
        self.noise = noise
        self.linear_layers = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.use_batch_norm = batch_norm
        self.input_batch_norm = input_batch_norm
        if self.input_batch_norm:
            self.input_batch_norm_layer = nn.BatchNorm1d(input_feature_n)

        for i in range(len(output_features_n)):
            if i == 0:
                lin = nn.Linear(input_feature_n, output_features_n[i])
            else:
                lin = nn.Linear(output_features_n[i - 1], output_features_n[i])
            lin.bias.data.uniform_(-self.noise, self.noise)
            lin.weight.data.uniform_(-self.noise, self.noise)  # ?!

            self.linear_layers.append(lin)
            self.relus.append(nn.ReLU())
            if self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(output_features_n[i]))

        self.single_out_row = single_out_row

    def forward(self, args):
        (G, x, tgt_out_rows) = args
        if self.input_batch_norm:
            x = self.input_batch_norm_layer(x.reshape(-1, self.input_feature_n))
            x = x.reshape(-1, self.MAX_N, self.input_feature_n)

        for l in range(len(self.linear_layers)):
            x = self.linear_layers[l](x)
            x = torch.bmm(G, x)
            x = self.relus[l](x)
            if self.use_batch_norm:
                x = self.batch_norms[l](x.reshape(-1, self.output_features_n[l]))
                x = x.reshape(-1, self.MAX_N, self.output_features_n[l])
        if self.single_out_row:
            return torch.stack([x[i, j] for i, j in enumerate(tgt_out_rows)])
        else:
            return x


class ResGraphModel(nn.Module):
    """
    Graphical resnet with batch norm structure
    """

    def __init__(
        self,
        MAX_N,
        input_feature_n,
        output_features_n,
        noise=1e-5,
        single_out_row=True,
        batch_norm=False,
        input_batch_norm=False,
        resnet=True,
    ):
        super(ResGraphModel, self).__init__()
        self.MAX_N = MAX_N
        self.input_feature_n = input_feature_n
        self.output_features_n = output_features_n
        self.noise = noise
        self.linear_layers = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.use_batch_norm = batch_norm
        self.input_batch_norm = input_batch_norm
        self.use_resnet = resnet

        if self.input_batch_norm:
            self.input_batch_norm_layer = nn.BatchNorm1d(input_feature_n)

        for i in range(len(output_features_n)):
            if i == 0:
                lin = nn.Linear(input_feature_n, output_features_n[i])
            else:
                lin = nn.Linear(output_features_n[i - 1], output_features_n[i])
            # nn.init.kaiming_uniform_(lin.weight.data, nonlinearity='relu')
            # nn.init.kaiming_uniform_(lin.weight.data, nonlinearity='relu')

            lin.bias.data.uniform_(-self.noise, self.noise)
            lin.weight.data.uniform_(-self.noise, self.noise)  # ?!

            self.linear_layers.append(lin)
            self.relus.append(nn.ReLU())
            if self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(output_features_n[i]))

        self.single_out_row = single_out_row

    def forward(self, args):
        (G, x, tgt_out_rows) = args
        if self.input_batch_norm:
            x = self.input_batch_norm_layer(x.reshape(-1, self.input_feature_n))
            x = x.reshape(-1, self.MAX_N, self.input_feature_n)

        for l in range(len(self.linear_layers)):
            x1 = torch.bmm(G, self.linear_layers[l](x))
            x2 = self.relus[l](x1)

            if x.shape == x2.shape and self.use_resnet:
                x3 = x2 + x
            else:
                x3 = x2
            if self.use_batch_norm:
                x = self.batch_norms[l](x3.reshape(-1, self.output_features_n[l]))
                x = x.reshape(-1, self.MAX_N, self.output_features_n[l])
            else:
                x = x3
        if self.single_out_row:
            return torch.stack([x[i, j] for i, j in enumerate(tgt_out_rows)])
        else:
            return x


def goodmax(x, dim):
    return torch.max(x, dim=dim)[0]


class GraphMatLayer(nn.Module):
    def __init__(
        self, C, P, GS=1, noise=1e-6, agg_func=None, dropout=0.0, use_bias=True
    ):
        """
        Pairwise layer -- takes a N x M x M x C matrix
        and turns it into a N x M x M x P matrix after
        multiplying with a graph matrix N x M x M

        if GS != 1 then there will be a per-graph-channel
        linear layer
        """
        super(GraphMatLayer, self).__init__()

        self.GS = GS
        self.noise = noise

        self.linlayers = nn.ModuleList()
        self.dropout = dropout
        self.dropout_layers = nn.ModuleList()
        for ll in range(GS):
            l = nn.Linear(C, P, bias=use_bias)
            if use_bias:
                l.bias.data.normal_(0.0, self.noise)
            l.weight.data.normal_(0.0, self.noise)  # ?!
            self.linlayers.append(l)
            if dropout > 0.0:
                self.dropout_layers.append(nn.Dropout(p=dropout))

        # self.r = nn.PReLU()
        self.r = nn.ReLU()
        self.agg_func = agg_func

    def forward(self, G, x):
        def apply_ll(i, x):
            y = self.linlayers[i](x)
            if self.dropout > 0:
                y = self.dropout_layers[i](y)
            return y

        multi_x = torch.stack([apply_ll(i, x) for i in range(self.GS)])
        # this is per-batch-element
        xout = torch.stack(
            [torch.matmul(G[i], multi_x[:, i]) for i in range(x.shape[0])]
        )

        x = self.r(xout)
        if self.agg_func is not None:
            x = self.agg_func(x, dim=1)
        return x


class GraphMatLayers(nn.Module):
    def __init__(
        self,
        input_feature_n,
        output_features_n,
        resnet=False,
        GS=1,
        norm=None,
        force_use_bias=False,
        noise=1e-5,
        agg_func=None,
        layer_class="GraphMatLayerFast",
        layer_config={},
    ):
        super(GraphMatLayers, self).__init__()

        self.gl = nn.ModuleList()
        self.resnet = resnet

        LayerClass = eval(layer_class)
        for li in range(len(output_features_n)):
            if li == 0:
                gl = LayerClass(
                    input_feature_n,
                    output_features_n[0],
                    noise=noise,
                    agg_func=agg_func,
                    GS=GS,
                    use_bias=not norm or force_use_bias,
                    **layer_config,
                )
            else:
                gl = LayerClass(
                    output_features_n[li - 1],
                    output_features_n[li],
                    noise=noise,
                    agg_func=agg_func,
                    GS=GS,
                    use_bias=not norm or force_use_bias,
                    **layer_config,
                )

            self.gl.append(gl)

        self.norm = norm
        if self.norm is not None:
            if self.norm == "batch":
                Nlayer = MaskedBatchNorm1d
            elif self.norm == "layer":
                Nlayer = MaskedLayerNorm1d
            self.bn = nn.ModuleList([Nlayer(f) for f in output_features_n])

    def forward(self, G, x, input_mask=None):
        for gi, gl in enumerate(self.gl):
            x2 = gl(G, x)
            if self.norm:
                x2 = self.bn[gi](
                    x2.reshape(-1, x2.shape[-1]), input_mask.reshape(-1)
                ).reshape(x2.shape)

            if self.resnet:
                if x.shape == x2.shape:
                    x3 = x2 + x
                else:
                    x3 = x2
            else:
                x3 = x2
            x = x3

        return x


class GraphMatHighwayLayers(nn.Module):
    def __init__(
        self,
        input_feature_n,
        output_features_n,
        resnet=False,
        GS=1,
        noise=1e-5,
        agg_func=None,
    ):
        super(GraphMatHighwayLayers, self).__init__()

        self.gl = nn.ModuleList()
        self.resnet = resnet

        for li in range(len(output_features_n)):
            if li == 0:
                gl = GraphMatLayer(
                    input_feature_n,
                    output_features_n[0],
                    noise=noise,
                    agg_func=agg_func,
                    GS=GS,
                )
            else:
                gl = GraphMatLayer(
                    output_features_n[li - 1],
                    output_features_n[li],
                    noise=noise,
                    agg_func=agg_func,
                    GS=GS,
                )

            self.gl.append(gl)

    def forward(self, G, x):
        highway_out = []
        for gl in self.gl:
            x2 = gl(G, x)
            if self.resnet:
                if x.shape == x2.shape:
                    x3 = x2 + x
                else:
                    x3 = x2
            else:
                x3 = x2
            x = x3
            highway_out.append(x2)

        return x, torch.stack(highway_out, -1)


def batch_diagonal_extract(x):
    BATCH_N, M, _, N = x.shape

    return torch.stack([x[:, i, i, :] for i in range(M)], dim=1)


class GraphMatModel(nn.Module):
    def __init__(
        self, g_feature_n, g_feature_out_n, resnet=True, noise=1e-5, GS=1, OUT_DIM=1
    ):
        """
        g_features_in : how many per-edge features
        g_features_out : how many per-edge features
        """
        super(GraphMatModel, self).__init__()

        self.gml = GraphMatLayers(
            g_feature_n, g_feature_out_n, resnet=resnet, noise=noise, GS=GS
        )

        self.lin_out = nn.Linear(g_feature_out_n[-1], OUT_DIM)
        # torch.nn.init.kaiming_uniform_(self.lin_out.weight.data, nonlinearity='relu')

    def forward(self, args):
        (G, x_G) = args

        G_features = self.gml(G, x_G)
        ## OLD WAY

        g_diag = batch_diagonal_extract(G_features)
        x_1 = self.lin_out(g_diag)

        return x_1


class GraphVertModel(nn.Module):
    def __init__(
        self,
        g_feature_n,
        g_feature_out_n=None,
        int_d=None,
        layer_n=None,
        resnet=True,
        init_noise=1e-5,
        agg_func=None,
        GS=1,
        OUT_DIM=1,
        input_batchnorm=False,
        out_std=False,
        resnet_out=False,
        resnet_blocks=(3,),
        resnet_d=128,
        graph_dropout=0.0,
        batchnorm=False,
        out_std_exp=False,
        force_lin_init=False,
    ):
        """ """
        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n

        super(GraphVertModel, self).__init__()
        self.gml = GraphMatLayers(
            g_feature_n,
            g_feature_out_n,
            resnet=resnet,
            noise=init_noise,
            agg_func=parse_agg_func(agg_func),
            batchnorm=batchnorm,
            GS=GS,
            dropout=graph_dropout,
        )

        if input_batchnorm:
            self.input_batchnorm = nn.BatchNorm1d(g_feature_n)
        else:
            self.input_batchnorm = None

        self.resnet_out = resnet_out
        if not resnet_out:
            self.lin_out = nn.Linear(g_feature_out_n[-1], OUT_DIM)
        else:
            self.lin_out = ResNetRegression(
                g_feature_out_n[-1],
                block_sizes=resnet_blocks,
                INT_D=resnet_d,
                FINAL_D=resnet_d,
                OUT_DIM=OUT_DIM,
            )

        self.out_std = out_std
        self.out_std_exp = False

        self.lin_out_std1 = nn.Linear(g_feature_out_n[-1], 128)
        self.lin_out_std2 = nn.Linear(128, OUT_DIM)

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        print("xavier init")
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(
        self, adj, vect_feat, input_mask, input_idx, return_g_features=False, **kwargs
    ):
        G = adj
        x_G = vect_feat

        BATCH_N, MAX_N, F_N = x_G.shape

        # if self.input_batchnorm is not None:
        #     x_G_flat = x_G.reshape(BATCH_N*MAX_N, F_N)
        #     x_G_out_flat = self.input_batchnorm(x_G_flat)
        #     x_G = x_G_out_flat.reshape(BATCH_N, MAX_N, F_N)

        G_features = self.gml(G, x_G)
        if return_g_features:
            return G_features

        g_squeeze = G_features.squeeze(1)
        g_squeeze_flat = g_squeeze.reshape(-1, G_features.shape[-1])

        if self.resnet_out:
            x_1 = self.lin_out(g_squeeze_flat).reshape(BATCH_N, MAX_N, -1)
        else:
            x_1 = self.lin_out(g_squeeze)

        if self.out_std:
            x_std = F.relu(self.lin_out_std1(g_squeeze))
            # if self.out_std_exp:
            #     x_1_std = F.exp(self.lin_out_std2(x_std))
            # else:
            x_1_std = F.relu(self.lin_out_std2(x_std))

            # g_2 = F.relu(self.lin_out_std(g_squeeze_flat))

            # x_1_std = g_2.reshape(BATCH_N, MAX_N, -1)

            return {"mu": x_1, "std": x_1_std}
        else:
            return {"mu": x_1, "std": 0.0 * x_1}


class GraphVertResOutModel(nn.Module):
    def __init__(
        self,
        g_feature_n,
        g_feature_out_n,
        resnet=True,
        noise=1e-5,
        agg_func=None,
        GS=1,
        OUT_DIM=1,
        batch_norm=False,
        out_std=False,
        force_lin_init=False,
    ):
        """ """
        super(GraphVertResOutModel, self).__init__()
        self.gml = GraphMatLayers(
            g_feature_n,
            g_feature_out_n,
            resnet=resnet,
            noise=noise,
            agg_func=agg_func,
            GS=GS,
        )

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(g_feature_n)
        else:
            self.batch_norm = None

        print("g_feature_out_n[-1]=", g_feature_out_n[-1])

        self.lin_out = ResNetRegression(
            g_feature_out_n[-1],
            block_sizes=[3],
            INT_D=128,
            FINAL_D=1024,
            OUT_DIM=OUT_DIM,
        )

        self.out_std = out_std

        # if out_std:
        #     self.lin_out_std = nn.Linear(g_feature_out_n[-1], 32)
        #     self.lin_out_std1 = nn.Linear(32, OUT_DIM)

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, noise)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, args):
        (G, x_G) = args

        BATCH_N, MAX_N, F_N = x_G.shape

        if self.batch_norm is not None:
            x_G_flat = x_G.reshape(BATCH_N * MAX_N, F_N)
            x_G_out_flat = self.batch_norm(x_G_flat)
            x_G = x_G_out_flat.reshape(BATCH_N, MAX_N, F_N)

        G_features = self.gml(G, x_G)

        g_squeeze = G_features.squeeze(1).reshape(-1, G_features.shape[-1])

        x_1 = self.lin_out(g_squeeze)

        return x_1.reshape(BATCH_N, MAX_N, -1)

        # if self.out_std:
        #     x_1_std = F.relu(self.lin_out_std(g_squeeze))
        #     x_1_std = F.relu(self.lin_out_std1(x_1_std))

        #     return x_1, x_1_std
        # else:
        #     return x_1


def parse_agg_func(agg_func):
    if isinstance(agg_func, str):
        if agg_func == "goodmax":
            return goodmax
        elif agg_func == "sum":
            return torch.sum
        elif agg_func == "mean":
            return torch.mean
        else:
            raise NotImplementedError()
    return agg_func


class GraphVertExtraLinModel(nn.Module):
    def __init__(
        self,
        g_feature_n,
        g_feature_out_n=None,
        resnet=True,
        int_d=None,
        layer_n=None,
        init_noise=1e-5,
        agg_func=None,
        GS=1,
        combine_in=False,
        OUT_DIM=1,
        force_lin_init=False,
        use_highway=False,
        use_graph_conv=True,
        extra_lin_int_d=128,
    ):
        """ """
        super(GraphVertExtraLinModel, self).__init__()

        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n
        print("g_feature_n=", g_feature_n)
        self.use_highway = use_highway
        if use_highway:
            self.gml = GraphMatHighwayLayers(
                g_feature_n,
                g_feature_out_n,
                resnet=resnet,
                noise=init_noise,
                agg_func=parse_agg_func(agg_func),
                GS=GS,
            )
        else:
            self.gml = GraphMatLayers(
                g_feature_n,
                g_feature_out_n,
                resnet=resnet,
                noise=init_noise,
                agg_func=parse_agg_func(agg_func),
                GS=GS,
            )

        self.combine_in = combine_in
        self.use_graph_conv = use_graph_conv
        lin_layer_feat = 0
        if use_graph_conv:
            lin_layer_feat += g_feature_out_n[-1]

        if combine_in:
            lin_layer_feat += g_feature_n
        if self.use_highway:
            lin_layer_feat += np.sum(g_feature_out_n)

        self.lin_out1 = nn.Linear(lin_layer_feat, extra_lin_int_d)
        self.lin_out2 = nn.Linear(extra_lin_int_d, OUT_DIM)

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, init_noise)
                    nn.init.constant_(m.bias, 0)

    def forward(self, args):
        G = args[0]
        x_G = args[1]
        if self.use_highway:
            G_features, G_highway = self.gml(G, x_G)
            G_highway_flatten = G_highway.reshape(
                G_highway.shape[0], G_highway.shape[1], -1
            )
        else:
            G_features = self.gml(G, x_G)

        g_squeeze = G_features.squeeze(1)
        out_feat = []
        if self.use_graph_conv:
            out_feat.append(g_squeeze)
        if self.combine_in:
            out_feat.append(x_G)
        if self.use_highway:
            out_feat.append(G_highway_flatten)

        lin_input = torch.cat(out_feat, -1)

        x_1 = self.lin_out1(lin_input)
        x_2 = self.lin_out2(F.relu(x_1))

        return x_2


class MSELogNormalLoss(nn.Module):
    def __init__(
        self, use_std_term=True, use_log1p=True, std_regularize=0.0, std_pow=2.0
    ):
        super(MSELogNormalLoss, self).__init__()
        self.use_std_term = use_std_term
        self.use_log1p = use_log1p
        self.std_regularize = std_regularize
        self.std_pow = std_pow

    def __call__(self, y, mu, std):
        if self.use_log1p:
            log = torch.log1p
        else:
            log = torch.log
        std = std + self.std_regularize

        std_term = -0.5 * log(2 * np.pi * std**2)
        log_pdf = -((y - mu) ** 2) / (2.0 * std**self.std_pow)
        if self.use_std_term:
            log_pdf += std_term

        return -log_pdf.mean()


def log_normal_nolog(y, mu, std):
    element_wise = -((y - mu) ** 2) / (2 * std**2) - std
    return element_wise


def log_student_t(y, mu, std, v=1.0):
    return -torch.log(1.0 + (y - mu) ** 2 / (v * std)) - std


def log_normal(y, mu, std):
    element_wise = -((y - mu) ** 2) / (2 * std**2) - torch.log(std)
    return element_wise


class MSECustomLoss(nn.Module):
    def __init__(
        self, use_std_term=True, use_log1p=True, std_regularize=0.0, std_pow=2.0
    ):
        super(MSECustomLoss, self).__init__()
        self.use_std_term = use_std_term
        self.use_log1p = use_log1p
        self.std_regularize = std_regularize
        self.std_pow = std_pow

    def __call__(self, y, mu, std):
        if self.use_log1p:
            log = torch.log1p
        else:
            log = torch.log
        std = std + self.std_regularize

        # std_term = -0.5 * log(2*np.pi * std**self.std_pow )
        # log_pdf = - (y-mu)**2/(2.0 * std **self.std_pow)

        # if self.use_std_term :
        #     log_pdf += std_term

        # return -log_pdf.mean()
        return -log_normal(y, mu, std).mean()


class MaskedMSELoss(nn.Module):
    """
    Masked mean squared error
    """

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mseloss = nn.MSELoss()

    def __call__(self, y, x, mask):
        x_masked = x[mask > 0].reshape(-1, 1)
        y_masked = y[mask > 0].reshape(-1, 1)
        return self.mseloss(x_masked, y_masked)


class MaskedMSSELoss(nn.Module):
    """
    Masked mean squared error
    """

    def __init__(self):
        super(MaskedMSSELoss, self).__init__()

    def __call__(self, y, x, mask):
        x_masked = x[mask > 0].reshape(-1, 1)
        y_masked = y[mask > 0].reshape(-1, 1)
        return ((x_masked - y_masked) ** 4).mean()


class MaskedMSEScaledLoss(nn.Module):
    """
    Masked mean squared error
    """

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mseloss = nn.MSELoss()

    def __call__(self, y, x, mask):
        x_masked = x[mask > 0].reshape(-1, 1)
        y_masked = y[mask > 0].reshape(-1, 1)
        return self.mseloss(x_masked, y_masked)


class NormUncertainLoss(nn.Module):
    """
    Masked uncertainty loss
    """

    def __init__(
        self,
        mu_scale=torch.Tensor([1.0]),
        std_scale=torch.Tensor([1.0]),
        use_std_term=True,
        use_log1p=False,
        std_regularize=0.0,
        std_pow=2.0,
        **kwargs,
    ):
        super(NormUncertainLoss, self).__init__()
        self.use_std_term = use_std_term
        self.use_log1p = use_log1p
        self.std_regularize = std_regularize
        self.std_pow = std_pow
        self.mu_scale = mu_scale
        self.std_scale = std_scale

    def __call__(self, pred, y, mask):
        ### NOTE pred is a tuple!
        mu, std = pred["mu"], pred["std"]

        if self.use_log1p:
            log = torch.log1p
        else:
            log = torch.log
        std = std + self.std_regularize

        y_scaled = y / self.mu_scale
        mu_scaled = mu / self.mu_scale
        std_scaled = std / self.std_scale

        y_scaled_masked = y_scaled[mask > 0].reshape(-1, 1)
        mu_scaled_masked = mu_scaled[mask > 0].reshape(-1, 1)
        std_scaled_masked = std_scaled[mask > 0].reshape(-1, 1)
        # return -log_normal_nolog(y_scaled_masked,
        #                          mu_scaled_masked,
        #                          std_scaled_masked).mean()
        return -log_normal_nolog(
            y_scaled_masked, mu_scaled_masked, std_scaled_masked
        ).mean()


class UncertainLoss(nn.Module):
    """
    simple uncertain loss
    """

    def __init__(
        self,
        mu_scale=1.0,
        std_scale=1.0,
        norm="l2",
        std_regularize=0.1,
        std_pow=2.0,
        std_weight=1.0,
        use_reg_log=False,
        **kwargs,
    ):
        super(UncertainLoss, self).__init__()
        self.mu_scale = mu_scale
        self.std_scale = std_scale
        self.std_regularize = std_regularize
        self.norm = norm

        if norm == "l2":
            self.loss = nn.MSELoss(reduction="none")
        elif norm == "huber":
            self.loss = nn.SmoothL1Loss(reduction="none")

        self.std_pow = std_pow
        self.std_weight = std_weight
        self.use_reg_log = use_reg_log

    def __call__(self, pred, y, mask, vert_mask):
        mu, std = pred["mu"], pred["std"]

        std = std + self.std_regularize

        y_scaled = y / self.mu_scale
        mu_scaled = mu / self.mu_scale
        std_scaled = std / self.std_scale

        y_scaled_masked = y_scaled[mask > 0].reshape(-1, 1)
        mu_scaled_masked = mu_scaled[mask > 0].reshape(-1, 1)
        std_scaled_masked = std_scaled[mask > 0].reshape(-1, 1)

        sm = std_scaled_masked**self.std_pow

        sml = std_scaled_masked
        if self.use_reg_log:
            sml = torch.log(sml)

        l = self.loss(y_scaled_masked, mu_scaled_masked) / (sm) + self.std_weight * sml
        return torch.mean(l)


class TukeyBiweight(nn.Module):
    """
    implementation of tukey's biweight loss

    """

    def __init__(self, c):
        self.c = c

    def __call__(self, true, pred):
        c = self.c

        r = true - pred
        r_abs = torch.abs(r)
        check = (r_abs <= c).float()

        sub_th = 1 - (1 - (r / c) ** 2) ** 3
        other = 1.0
        # print(true.shape, pred.shape, sub_th
        result = sub_th * check + 1.0 * (1 - check)
        return torch.mean(result * c**2 / 6.0)


class NoUncertainLoss(nn.Module):
    """ """

    def __init__(self, norm="l2", scale=1.0, **kwargs):
        super(NoUncertainLoss, self).__init__()
        if norm == "l2":
            self.loss = nn.MSELoss()
        elif norm == "huber":
            self.loss = nn.SmoothL1Loss()
        elif "tukeybw" in norm:
            c = float(norm.split("-")[1])
            self.loss = TukeyBiweight(c)

        self.scale = scale

    def __call__(
        self, res, vert_pred, vert_pred_mask, edge_pred, edge_pred_mask, vert_mask
    ):
        mu = res["shift_mu"]
        mask = vert_pred_mask

        assert torch.sum(mask) > 0
        y_masked = vert_pred[mask > 0].reshape(-1, 1) * self.scale
        mu_masked = mu[mask > 0].reshape(-1, 1) * self.scale

        return self.loss(y_masked, mu_masked)


class SimpleLoss(nn.Module):
    """ """

    def __init__(self, norm="l2", scale=1.0, **kwargs):
        super(SimpleLoss, self).__init__()
        if norm == "l2":
            self.loss = nn.MSELoss()
        elif norm == "huber":
            self.loss = nn.SmoothL1Loss()
        elif "tukeybw" in norm:
            c = float(norm.split("-")[1])
            self.loss = TukeyBiweight(c)

        self.scale = scale

    def __call__(
        self,
        pred,
        vert_pred,
        vert_pred_mask,
        edge_pred,
        edge_pred_mask,
        ## ADD OTHJERS
        vert_mask,
    ):
        mu = pred["mu"]  ## FIXME FOR VERT

        assert torch.sum(vert_pred_mask) > 0

        y_masked = vert_pred[vert_pred_mask > 0].reshape(-1, 1) * self.scale
        mu_masked = mu[vert_pred_mask > 0].reshape(-1, 1) * self.scale

        return self.loss(y_masked, mu_masked)


class GraphMatLayerFast(nn.Module):
    def __init__(
        self,
        C,
        P,
        GS=1,
        noise=1e-6,
        agg_func=None,
        dropout=False,
        use_bias=False,
    ):
        """
        Pairwise layer -- takes a N x M x M x C matrix
        and turns it into a N x M x M x P matrix after
        multiplying with a graph matrix N x M x M

        if GS != 1 then there will be a per-graph-channel
        linear layer
        """
        super(GraphMatLayerFast, self).__init__()

        self.GS = GS
        self.noise = noise

        self.linlayers = nn.ModuleList()
        for ll in range(GS):
            l = nn.Linear(C, P, bias=use_bias)
            if self.noise == 0:
                if use_bias:
                    l.bias.data.normal_(0.0, 1e-4)
                torch.nn.init.xavier_uniform_(l.weight)
            else:
                if use_bias:
                    l.bias.data.normal_(0.0, self.noise)
                l.weight.data.normal_(0.0, self.noise)  # ?!
            self.linlayers.append(l)

        # self.r = nn.PReLU()
        self.r = nn.LeakyReLU()
        self.agg_func = agg_func

    def forward(self, G, x):
        BATCH_N, CHAN_N, MAX_N, _ = G.shape

        def apply_ll(i, x):
            y = self.linlayers[i](x)
            return y

        multi_x = torch.stack([apply_ll(i, x) for i in range(self.GS)], 0)
        xout = torch.einsum("ijkl,jilm->jikm", [G, multi_x])
        xout = self.r(xout)
        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)
        return xout


class GraphMatLayerFastSCM(nn.Module):
    def __init__(self, C, P, GS=1, noise=1e-6, agg_func=None, nonlin="relu"):
        """
        Pairwise layer -- takes a N x M x M x C matrix
        and turns it into a N x M x M x P matrix after
        multiplying with a graph matrix N x M x M

        if GS != 1 then there will be a per-graph-channel
        linear layer
        """
        super(GraphMatLayerFastSCM, self).__init__()

        self.GS = GS
        self.noise = noise

        self.linlayers = nn.ModuleList()
        for ll in range(GS):
            l = nn.Linear(C, P)
            if self.noise == 0:
                l.bias.data.normal_(0.0, 1e-4)
                torch.nn.init.xavier_uniform_(l.weight)
            else:
                l.bias.data.normal_(0.0, self.noise)
                l.weight.data.normal_(0.0, self.noise)  # ?!
            self.linlayers.append(l)

        # self.r = nn.PReLU()
        if nonlin == "relu":
            self.r = nn.ReLU()
        elif nonlin == "prelu":
            self.r = nn.PReLU()
        elif nonlin == "selu":
            self.r = nn.SELU()
        else:
            raise ValueError(nonlin)
        self.agg_func = agg_func

    def forward(self, G, x):
        BATCH_N, CHAN_N, MAX_N, _ = G.shape

        def apply_ll(i, x):
            y = self.linlayers[i](x)
            return y

        multi_x = torch.stack([apply_ll(i, x) for i in range(self.GS)], 0)
        xout = torch.einsum("ijkl,jilm->jikm", [G, multi_x])

        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)
        return self.r(xout)


class SGCModel(nn.Module):
    def __init__(
        self,
        g_feature_n=4,
        int_d=128,
        OUT_DIM=1,
        GS=4,
        agg_func="goodmax",
        force_lin_init=False,
        init_noise=0.1,
        out_std=False,
        resnet_out=False,
        resnet_blocks=(3,),
        resnet_d=128,
        input_batchnorm=False,
        gml_nonlin="selu",
        **kwargs,
    ):
        """
        SGC-esque model form Simplifying Graph Convolutional Networks


        """

        super(SGCModel, self).__init__()

        self.gml = GraphMatLayerFastSCM(
            g_feature_n,
            int_d,
            GS=GS,
            agg_func=parse_agg_func(agg_func),
            nonlin=gml_nonlin,
        )

        self.resnet_out = resnet_out

        self.lin_out_mu = nn.Linear(int_d, OUT_DIM)
        self.lin_out_std = nn.Linear(int_d, OUT_DIM)

        if not resnet_out:
            self.lin_out_first = nn.Linear(int_d, int_d)
        else:
            self.lin_out_first = ResNetRegression(
                int_d,
                block_sizes=resnet_blocks,
                INT_D=resnet_d,
                FINAL_D=resnet_d,
                OUT_DIM=int_d,
            )

        if input_batchnorm:
            self.input_batchnorm = MaskedBatchNorm1d(g_feature_n)
        else:
            self.input_batchnorm = None

        self.out_std = out_std

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        print("xavier init")
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(
        self, adj, vect_feat, input_mask, input_idx, return_g_features=False, **kwargs
    ):
        G = adj
        x_G = vect_feat

        BATCH_N, MAX_N, F_N = vect_feat.shape

        if self.input_batchnorm is not None:
            vect_feat_flat = vect_feat.reshape(BATCH_N * MAX_N, F_N)
            input_mask_flat = input_mask.reshape(BATCH_N * MAX_N)
            vect_feat_out_flat = self.input_batchnorm(vect_feat_flat, input_mask_flat)
            vect_feat = vect_feat_out_flat.reshape(BATCH_N, MAX_N, F_N)

        G_features = self.gml(G, vect_feat)

        g_squeeze = G_features.squeeze(1)
        g_squeeze_flat = g_squeeze.reshape(-1, G_features.shape[-1])

        if self.resnet_out:
            x = F.relu(self.lin_out_first(g_squeeze_flat).reshape(BATCH_N, MAX_N, -1))
        else:
            x = F.relu(self.lin_out_first(g_squeeze))

        x_mu = self.lin_out_mu(x)
        x_std = F.relu(self.lin_out_std(x))

        if self.out_std:
            return {"mu": x_mu, "std": x_std}
        else:
            return {"mu": x_mu, "std": 0.0 * x_mu}


class SGCModelNoAgg(nn.Module):
    def __init__(
        self,
        g_feature_n=4,
        int_d=128,
        OUT_DIM=1,
        GS=4,  # agg_func = 'goodmax',
        force_lin_init=False,
        init_noise=0.1,
        out_std=False,
        resnet_out=False,
        resnet_blocks=(3,),
        resnet_d=128,
        **kwargs,
    ):
        """
        SGC-esque model form Simplifying Graph Convolutional Networks


        """

        super(SGCModelNoAgg, self).__init__()

        self.gml = GraphMatLayerFast(g_feature_n, int_d, GS=GS, agg_func=None)

        self.resnet_out = resnet_out

        self.lin_out_mu = nn.Linear(int_d, OUT_DIM)
        self.lin_out_std = nn.Linear(int_d, OUT_DIM)

        if not resnet_out:
            self.lin_out_first = nn.Linear(int_d * GS, int_d)
        else:
            self.lin_out_first = ResNetRegression(
                int_d,
                block_sizes=resnet_blocks,
                INT_D=resnet_d,
                FINAL_D=resnet_d,
                OUT_DIM=int_d,
            )

        self.out_std = out_std

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        print("xavier init")
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, args, return_g_features=False):
        (G, x_G) = args

        BATCH_N, MAX_N, F_N = x_G.shape

        G_features = self.gml(G, x_G)

        G_features = G_features.permute(1, 2, 3, 0)
        G_features = G_features.reshape(BATCH_N, MAX_N, -1)

        g_squeeze_flat = G_features.reshape(-1, G_features.shape[-1])

        if self.resnet_out:
            raise ValueError()
            x = F.relu(self.lin_out_first(g_squeeze_flat).reshape(BATCH_N, MAX_N, -1))
        else:
            x = F.relu(self.lin_out_first(G_features))

        x_mu = self.lin_out_mu(x)
        x_std = F.relu(self.lin_out_std(x))

        if self.out_std:
            return {"mu": x_mu, "std": x_std}
        else:
            return {"mu": x_mu, "std": 0.0 * x_mu}


class PredVertOnly(nn.Module):
    def __init__(
        self,
        g_feature_n=4,
        int_d=128,
        OUT_DIM=1,
        resnet_blocks=(3,),
        resnet_d=128,
        resnet_out=True,
        out_std=True,
        use_batchnorm=False,
        input_batchnorm=True,
        force_lin_init=True,
        init_noise=0.0,
        **kwargs,
    ):
        """
        Just a simple model that predicts directly from the vertex
        features and ignores the graph properties
        """

        super(PredVertOnly, self).__init__()

        self.resnet_out = resnet_out
        self.g_feature_n = g_feature_n

        self.lin_out_mu = nn.Linear(int_d, OUT_DIM)
        self.lin_out_std = nn.Linear(int_d, OUT_DIM)

        self.use_batchnorm = use_batchnorm
        if not resnet_out:
            self.lin_out_first = nn.Linear(int_d * GS, int_d)
        else:
            if use_batchnorm:
                self.lin_out_first = ResNetRegressionMaskedBN(
                    int_d,
                    block_sizes=resnet_blocks,
                    INT_D=resnet_d,
                    FINAL_D=resnet_d,
                    OUT_DIM=int_d,
                )

            else:
                self.lin_out_first = ResNetRegression(
                    int_d,
                    block_sizes=resnet_blocks,
                    INT_D=resnet_d,
                    FINAL_D=resnet_d,
                    OUT_DIM=int_d,
                )
        self.pad_layer = nn.ConstantPad1d((0, int_d - g_feature_n), 0.0)

        if input_batchnorm:
            self.input_batchnorm = MaskedBatchNorm1d(g_feature_n)
        else:
            self.input_batchnorm = None

        self.out_std = out_std

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        print("xavier init")
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(
        self, adj, vect_feat, input_mask, input_idx, return_g_features=False, **kwargs
    ):
        G = adj

        BATCH_N, MAX_N, F_N = vect_feat.shape
        input_mask_flat = input_mask.reshape(BATCH_N * MAX_N)

        if self.input_batchnorm is not None:
            vect_feat_flat = vect_feat.reshape(BATCH_N * MAX_N, F_N)
            vect_feat_out_flat = self.input_batchnorm(vect_feat_flat, input_mask_flat)
            vect_feat = vect_feat_out_flat.reshape(BATCH_N, MAX_N, F_N)

        G_features = self.pad_layer(vect_feat)

        g_squeeze_flat = G_features.reshape(-1, G_features.shape[-1])

        if self.resnet_out:
            if self.use_batchnorm:
                x = F.relu(
                    self.lin_out_first(g_squeeze_flat, input_mask_flat).reshape(
                        BATCH_N, MAX_N, -1
                    )
                )

            else:
                x = F.relu(
                    self.lin_out_first(g_squeeze_flat).reshape(BATCH_N, MAX_N, -1)
                )
        else:
            x = F.relu(self.lin_out_first(G_features))

        x_mu = self.lin_out_mu(x)
        x_std = F.relu(self.lin_out_std(x))

        if self.out_std:
            return {"mu": x_mu, "std": x_std}
        else:
            return {"mu": x_mu, "std": 0.0 * x_mu}


class RelNetFromS2S(nn.Module):
    def __init__(
        self,
        vert_f_in,
        edge_f_in,
        MAX_N,
        layer_n,
        internal_d_vert,
        internal_d_edge,
        init_noise=0.01,
        force_lin_init=False,
        dim_out=4,
        final_d_out=64,
        force_bias_zero=True,
        bilinear_vv=False,
        gru_vv=False,
        gru_ab=True,
        layer_use_bias=True,
        edge_mat_norm=False,
        force_edge_zero=False,
        v_resnet_every=1,
        e_resnet_every=1,
        # softmax_out = False,
        OUT_DIM=1,
        pos_out=False,
        out_std=False,
        resnet_out=False,
        resnet_blocks=(3,),
        resnet_d=128,
    ):
        """
        Relational net stolen from S2S
        """

        super(RelNetFromS2S, self).__init__()

        self.MAX_N = MAX_N
        self.vert_f_in = vert_f_in
        self.edge_f_in = edge_f_in

        self.dim_out = dim_out
        self.internal_d_vert = internal_d_vert
        self.layer_n = layer_n
        self.edge_mat_norm = edge_mat_norm
        self.force_edge_zero = force_edge_zero

        self.lin_e_layers = nn.ModuleList(
            [
                nn.Linear(internal_d_vert, internal_d_vert, bias=layer_use_bias)
                for _ in range(self.layer_n)
            ]
        )

        self.lin_v_layers = nn.ModuleList(
            [
                nn.Linear(internal_d_vert, internal_d_vert, bias=layer_use_bias)
                for _ in range(self.layer_n)
            ]
        )

        self.input_v_bn = nn.BatchNorm1d(vert_f_in)
        self.input_e_bn = nn.BatchNorm1d(edge_f_in)

        self.bilinear_vv = bilinear_vv
        self.gru_vv = gru_vv
        self.gru_ab = gru_ab

        if self.bilinear_vv:
            self.lin_vv_layers = nn.ModuleList(
                [
                    nn.Bilinear(
                        internal_d_vert,
                        internal_d_vert,
                        internal_d_vert,
                        bias=layer_use_bias,
                    )
                    for _ in range(self.layer_n)
                ]
            )
        elif self.gru_vv:
            self.lin_vv_layers = nn.ModuleList(
                [
                    nn.GRUCell(internal_d_vert, internal_d_vert, bias=layer_use_bias)
                    for _ in range(self.layer_n)
                ]
            )
        else:
            self.lin_vv_layers = nn.ModuleList(
                [
                    nn.Linear(
                        internal_d_vert + internal_d_vert,
                        internal_d_vert,
                        bias=layer_use_bias,
                    )
                    for _ in range(self.layer_n)
                ]
            )

        self.bn_v_layers = nn.ModuleList(
            [nn.BatchNorm1d(internal_d_vert) for _ in range(self.layer_n)]
        )

        self.bn_e_layers = nn.ModuleList(
            [nn.BatchNorm1d(internal_d_vert) for _ in range(self.layer_n)]
        )

        # self.per_v_l = nn.Linear(internal_d_vert, final_d_out)

        # self.per_v_l_1 = nn.Linear(final_d_out, final_d_out)

        self.resnet_out = resnet_out
        if not resnet_out:
            self.lin_out = nn.Linear(internal_d_vert, final_d_out)
        else:
            self.lin_out = ResNetRegression(
                internal_d_vert,
                block_sizes=resnet_blocks,
                INT_D=resnet_d,
                FINAL_D=final_d_out,
                OUT_DIM=final_d_out,
            )

        self.OUT_DIM = OUT_DIM

        self.init_noise = init_noise

        # self.triu_idx = torch.Tensor(triu_indices_flat(MAX_N, k=1)).long()

        self.v_resnet_every_n_layers = v_resnet_every
        self.e_resnet_every_n_layers = e_resnet_every

        # self.softmax_out = softmax_out

        self.pos_out = pos_out

        self.out_std = out_std

        self.lin_out_mu = nn.Linear(final_d_out, OUT_DIM)
        self.lin_out_std = nn.Linear(final_d_out, OUT_DIM)

        if force_lin_init:
            self.force_init(init_noise, force_bias_zero)

    def force_init(self, init_noise=None, force_bias_zero=True):
        if init_noise is None:
            init_noise = self.init_noise
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_noise < 1e-12:
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.normal_(m.weight, 0, init_noise)
                if force_bias_zero:
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    # def forward(self, v_in, e_in): # , graph_conn_in, out_mask=None):
    def forward(self, args, return_g_features=False):
        (e_in, v_in) = args

        """
        Remember input is 


        output is:
        [BATCH_N, FLATTEN_LENGHED_N, LABEL_LEVELS, M]
        """

        # print(v.shape, e_in.shape, graph_conn_in.shape)
        # torch.Size([16, 32, 81]) torch.Size([16, 32, 32, 19]) torch.Size([16, 32, 32, 3])
        BATCH_N = v_in.shape[0]
        MAX_N = v_in.shape[1]

        e_in = e_in.permute(0, 2, 3, 1)

        # print('v.shape=', v.shape, 'e_in.shape=', e_in.shape)

        def v_osum(v):
            return v.unsqueeze(1) + v.unsqueeze(2)

        def last_bn(layer, x):
            init_shape = x.shape

            x_flat = x.reshape(-1, init_shape[-1])
            x_bn = layer(x_flat)
            return x_bn.reshape(init_shape)

        def combine_vv(li, v1, v2):
            if self.bilinear_vv:
                return self.lin_vv_layers[li](v1, v2)
            elif self.gru_vv:
                if self.gru_ab:
                    a, b = v1, v2
                else:
                    a, b = v2, v1
                return self.lin_vv_layers[li](
                    a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
                ).reshape(a.shape)
            else:
                return self.lin_vv_layers[li](torch.cat([v1, v2], dim=-1))

        f1 = torch.relu
        f2 = torch.relu

        ### DEBUG FORCE TO ZERO
        if self.force_edge_zero:
            e_in[:] = 0

        if self.edge_mat_norm:
            e_in = batch_mat_chan_norm(e_in)

        def resnet_mod(i, k):
            if k > 0:
                if (i % k) == k - 1:
                    return True
            return False

        v_in_bn = last_bn(self.input_v_bn, v_in)
        e_in_bn = last_bn(self.input_e_bn, e_in)

        v = F.pad(v_in_bn, (0, self.internal_d_vert - v_in_bn.shape[-1]), "constant", 0)

        e = F.pad(e_in_bn, (0, self.internal_d_vert - e_in_bn.shape[-1]), "constant", 0)

        # v_flat, _ = self.v_global_l(v).max(dim=1)
        # v_flat = v_flat.unsqueeze(1)

        for li in range(self.layer_n):
            v_in = v
            e_in = e

            v_1 = self.lin_v_layers[li](v_in)
            e_1 = self.lin_e_layers[li](e_in)

            e_out = f1(e_1 + v_osum(v_1))
            e_out = last_bn(self.bn_e_layers[li], e_out)
            v_e = goodmax(e_out, 1)

            v_out = f2(combine_vv(li, v_in, v_e))
            v_out = last_bn(self.bn_v_layers[li], v_out)

            if resnet_mod(li, self.v_resnet_every_n_layers):
                v_out = v_out + v_in

            if resnet_mod(li, self.e_resnet_every_n_layers):
                e_out = e_out + e_in
            v = v_out
            e = e_out

        v_new = torch.relu(v)
        # e_new = e #

        # print("output: v.shape=", v.shape,
        #      "e.shape=", e.shape)

        # print("v_new.shape=", v_new.shape)
        # v_est_int = torch.relu(self.per_v_l(v_new))
        # v_est_int = torch.relu(self.per_v_l_1(v_est_int))

        v_squeeze_flat = v_new.reshape(-1, v_new.shape[-1])

        if self.resnet_out:
            x_1 = self.lin_out(v_squeeze_flat).reshape(BATCH_N, MAX_N, -1)
        else:
            x_1 = self.lin_out(v_new)

        v_est_int = torch.relu(x_1)

        x_mu = self.lin_out_mu(v_est_int)
        x_std = F.relu(self.lin_out_std(v_est_int))

        if self.out_std:
            return {"mu": x_mu, "std": x_std}
        else:
            return {"mu": x_mu, "std": 0.0 * x_mu}

        # ##multi_e_out = multi_e_out.squeeze(-2)

        # a_flat = e_est.reshape(BATCH_N, -1, self.dim_out, self.OUT_DIM)
        # #print("a_flat.shape=", a_flat.shape)
        # a_triu_flat = a_flat[:, self.triu_idx, :, :]

        # if self.logsoftmax_out:
        #     SOFTMAX_OFFSET = -1e6
        #     if out_mask is not None:
        #         out_mask_offset = SOFTMAX_OFFSET * (1-out_mask.unsqueeze(-1).unsqueeze(-1))
        #         a_triu_flat += out_mask_offset
        #     a_triu_flatter = a_triu_flat.reshape(BATCH_N, -1, 1)
        #     if self.logsoftmax_out:
        #         a_nonlin = F.log_softmax(a_triu_flatter, dim=1)
        #     elif self.softmax_out:
        #         a_nonlin = F.softmax(a_triu_flatter, dim=1)
        #     else:
        #         raise ValueError()

        #     a_nonlin = a_nonlin.reshape(BATCH_N, -1, self.dim_out, 1)
        # else:

        #     a_nonlin = a_triu_flat

        # if self.pos_out:
        #     a_nonlin = F.relu(a_nonlin)

        # return a_nonlin


class GraphVertModelExtraVertInput(nn.Module):
    def __init__(
        self,
        g_feature_n,
        g_feature_out_n=None,
        int_d=None,
        layer_n=None,
        resnet=True,
        init_noise=1e-5,
        agg_func=None,
        GS=1,
        OUT_DIM=1,
        input_batchnorm=False,
        out_std=False,
        resnet_out=False,
        resnet_blocks=(3,),
        resnet_d=128,
        graph_dropout=0.0,
        batchnorm=False,
        out_std_exp=False,
        force_lin_init=False,
        extra_vert_in_d=0,
    ):
        """ """
        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n

        super(GraphVertModelExtraVertInput, self).__init__()
        self.gml = GraphMatLayers(
            g_feature_n,
            g_feature_out_n,
            resnet=resnet,
            noise=init_noise,
            agg_func=parse_agg_func(agg_func),
            batchnorm=batchnorm,
            GS=GS,
            dropout=graph_dropout,
        )

        if input_batchnorm:
            self.input_batchnorm = nn.BatchNorm1d(g_feature_n)

        else:
            self.input_batchnorm = None

        if extra_vert_in_d > 0:
            self.input_extra_batchnorm = torch.nn.BatchNorm1d(extra_vert_in_d)
        else:
            self.input_extra_batchnorm = None

        self.resnet_out = resnet_out
        if not resnet_out:
            self.lin_out = nn.Linear(g_feature_out_n[-1], OUT_DIM)
        else:
            self.lin_out = ResNetRegression(
                g_feature_out_n[-1] + extra_vert_in_d,
                block_sizes=resnet_blocks,
                INT_D=resnet_d,
                FINAL_D=resnet_d,
                OUT_DIM=OUT_DIM,
            )

        self.out_std = out_std
        self.out_std_exp = False

        self.lin_out_std1 = nn.Linear(g_feature_out_n[-1], 128)
        self.lin_out_std2 = nn.Linear(128, OUT_DIM)

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        print("xavier init")
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, args, return_g_features=False):
        (G, x_G, extra_vert_arg) = args

        BATCH_N, MAX_N, F_N = x_G.shape

        # if self.input_batchnorm is not None:
        #     x_G_flat = x_G.reshape(BATCH_N*MAX_N, F_N)
        #     x_G_out_flat = self.input_batchnorm(x_G_flat)
        #     x_G = x_G_out_flat.reshape(BATCH_N, MAX_N, F_N)

        G_features = self.gml(G, x_G)
        if return_g_features:
            return G_features

        g_squeeze = G_features.squeeze(1)
        g_squeeze_flat = g_squeeze.reshape(-1, G_features.shape[-1])

        extra_vert_arg_flat = extra_vert_arg.reshape(BATCH_N * MAX_N, -1)

        if self.resnet_out:
            combined = torch.cat([g_squeeze_flat, extra_vert_arg_flat], -1)
            x_1 = self.lin_out(combined).reshape(BATCH_N, MAX_N, -1)
        else:
            raise ValueError("not concatenating yet")
            x_1 = self.lin_out(g_squeeze)

        if self.out_std:
            x_std = F.relu(self.lin_out_std1(g_squeeze))
            x_1_std = F.relu(self.lin_out_std2(x_std))

            return {"mu": x_1, "std": x_1_std}
        else:
            return {"mu": x_1, "std": 0.0 * x_1}


class GraphVertModelMaskedBN(nn.Module):
    def __init__(
        self,
        g_feature_n,
        g_feature_out_n=None,
        int_d=None,
        layer_n=None,
        resnet=True,
        init_noise=1e-5,
        agg_func=None,
        GS=1,
        OUT_DIM=1,
        input_batchnorm=False,
        out_std=False,
        resnet_out=False,
        resnet_blocks=(3,),
        resnet_d=128,
        graph_dropout=0.0,
        batchnorm=False,
        out_std_exp=False,
        force_lin_init=False,
    ):
        """ """
        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n

        super(GraphVertModelMaskedBN, self).__init__()
        self.gml = GraphMatLayers(
            g_feature_n,
            g_feature_out_n,
            resnet=resnet,
            noise=init_noise,
            agg_func=parse_agg_func(agg_func),
            batchnorm=batchnorm,
            GS=GS,
            dropout=graph_dropout,
        )

        if input_batchnorm:
            self.input_batchnorm = MaskedBatchNorm1d(g_feature_n)
        else:
            self.input_batchnorm = None

        self.resnet_out = resnet_out
        if not resnet_out:
            self.lin_out = nn.Linear(g_feature_out_n[-1], OUT_DIM)
        else:
            self.lin_out = ResNetRegression(
                g_feature_out_n[-1],
                block_sizes=resnet_blocks,
                INT_D=resnet_d,
                FINAL_D=resnet_d,
                OUT_DIM=OUT_DIM,
            )

        self.out_std = out_std
        self.out_std_exp = False

        self.lin_out_std1 = nn.Linear(g_feature_out_n[-1], 128)
        self.lin_out_std2 = nn.Linear(128, OUT_DIM)

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        print("xavier init")
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(
        self, adj, vect_feat, input_mask, input_idx, return_g_features=False, **kwargs
    ):
        G = adj
        x_G = vect_feat

        BATCH_N, MAX_N, F_N = vect_feat.shape

        if self.input_batchnorm is not None:
            x_G_flat = x_G.reshape(BATCH_N * MAX_N, F_N)
            input_mask_flat = input_mask.reshape(BATCH_N * MAX_N)
            x_G_out_flat = self.input_batchnorm(x_G_flat, input_mask_flat)
            x_G = x_G_out_flat.reshape(BATCH_N, MAX_N, F_N)

        G_features = self.gml(G, x_G, input_mask)
        if return_g_features:
            return G_features

        g_squeeze = G_features.squeeze(1)
        g_squeeze_flat = g_squeeze.reshape(-1, G_features.shape[-1])

        if self.resnet_out:
            x_1 = self.lin_out(g_squeeze_flat).reshape(BATCH_N, MAX_N, -1)
        else:
            x_1 = self.lin_out(g_squeeze)

        if self.out_std:
            x_std = F.relu(self.lin_out_std1(g_squeeze))
            # if self.out_std_exp:
            #     x_1_std = F.exp(self.lin_out_std2(x_std))
            # else:
            x_1_std = F.relu(self.lin_out_std2(x_std))

            # g_2 = F.relu(self.lin_out_std(g_squeeze_flat))

            # x_1_std = g_2.reshape(BATCH_N, MAX_N, -1)

            return {"mu": x_1, "std": x_1_std}
        else:
            return {"mu": x_1, "std": 0.0 * x_1}


class GraphVertModelMaskedBN2(nn.Module):
    def __init__(
        self,
        g_feature_n,
        g_feature_out_n=None,
        int_d=None,
        layer_n=None,
        resnet=True,
        init_noise=1e-5,
        agg_func=None,
        GS=1,
        OUT_DIM=1,
        input_batchnorm=False,
        out_std=False,
        resnet_out=False,
        resnet_blocks=(3,),
        resnet_d=128,
        graph_dropout=0.0,
        batchnorm=False,
        out_std_exp=False,
        force_lin_init=False,
    ):
        """ """
        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n

        super(GraphVertModelMaskedBN2, self).__init__()
        self.gml = GraphMatLayers(
            g_feature_n,
            g_feature_out_n,
            resnet=resnet,
            noise=init_noise,
            agg_func=parse_agg_func(agg_func),
            batchnorm=batchnorm,
            GS=GS,
            dropout=graph_dropout,
        )

        if input_batchnorm:
            self.input_batchnorm = MaskedBatchNorm1d(g_feature_n)
        else:
            self.input_batchnorm = None

        self.resnet_out = resnet_out
        if not resnet_out:
            self.lin_out_mu = nn.Linear(g_feature_out_n[-1], OUT_DIM)
            self.lin_out_std = nn.Linear(g_feature_out_n[-1], OUT_DIM)
        else:
            self.lin_out_mu = ResNetRegression(
                g_feature_out_n[-1],
                block_sizes=resnet_blocks,
                INT_D=resnet_d,
                FINAL_D=resnet_d,
                OUT_DIM=OUT_DIM,
            )

            self.lin_out_std = ResNetRegression(
                g_feature_out_n[-1],
                block_sizes=resnet_blocks,
                INT_D=resnet_d,
                FINAL_D=resnet_d,
                OUT_DIM=OUT_DIM,
            )

        self.out_std = out_std
        self.out_std_exp = False

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        print("xavier init")
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, args, return_g_features=False):
        (G, x_G, input_mask) = args

        BATCH_N, MAX_N, F_N = x_G.shape

        if self.input_batchnorm is not None:
            x_G_flat = x_G.reshape(BATCH_N * MAX_N, F_N)
            input_mask_flat = input_mask.reshape(BATCH_N * MAX_N)
            x_G_out_flat = self.input_batchnorm(x_G_flat, input_mask_flat)
            x_G = x_G_out_flat.reshape(BATCH_N, MAX_N, F_N)

        G_features = self.gml(G, x_G, input_mask)
        if return_g_features:
            return G_features

        g_squeeze = G_features.squeeze(1)
        g_squeeze_flat = g_squeeze.reshape(-1, G_features.shape[-1])

        if self.resnet_out:
            x_1 = self.lin_out_mu(g_squeeze_flat).reshape(BATCH_N, MAX_N, -1)
            x_std = F.relu(self.lin_out_std(g_squeeze_flat).reshape(BATCH_N, MAX_N, -1))
        else:
            x_1 = self.lin_out_mu(g_squeeze)
            x_std = F.relu(self.lin_out_std(g_squeeze))

        if self.out_std:
            return {"mu": x_1, "std": x_std}
        else:
            return {"mu": x_1, "std": 0.0 * x_1}


class GraphVertModelBootstrap(nn.Module):
    def __init__(
        self,
        g_feature_n,
        g_feature_out_n=None,
        int_d=None,
        layer_n=None,
        mixture_n=5,
        resnet=True,
        init_noise=1e-5,
        agg_func=None,
        GS=1,
        OUT_DIM=1,
        input_batchnorm=False,
        out_std=False,
        resnet_out=False,
        resnet_blocks=(3,),
        resnet_d=128,
        graph_dropout=0.0,
        norm="batch",
        out_std_exp=False,
        force_lin_init=False,
        use_random_subsets=True,
    ):
        """ """
        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n
        print("g_feature_out_n=", g_feature_out_n)

        super(GraphVertModelBootstrap, self).__init__()
        self.gml = GraphMatLayers(
            g_feature_n,
            g_feature_out_n,
            resnet=resnet,
            noise=init_noise,
            agg_func=parse_agg_func(agg_func),
            norm=norm,
            GS=GS,
        )

        if input_batchnorm:
            self.input_batchnorm = MaskedBatchNorm1d(g_feature_n)
        else:
            self.input_batchnorm = None

        self.resnet_out = resnet_out
        if not resnet_out:
            self.mix_out = nn.ModuleList(
                [nn.Linear(g_feature_out_n[-1], OUT_DIM) for _ in range(mixture_n)]
            )
        else:
            self.mix_out = nn.ModuleList(
                [
                    ResNetRegression(
                        g_feature_out_n[-1],
                        block_sizes=resnet_blocks,
                        INT_D=resnet_d,
                        FINAL_D=resnet_d,
                        OUT_DIM=OUT_DIM,
                    )
                    for _ in range(mixture_n)
                ]
            )

        self.out_std = out_std
        self.out_std_exp = False

        self.use_random_subsets = use_random_subsets

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        print("xavier init")
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(
        self, adj, vect_feat, input_mask, input_idx, return_g_features=False, **kwargs
    ):
        G = adj

        BATCH_N, MAX_N, F_N = vect_feat.shape

        if self.input_batchnorm is not None:
            vect_feat_flat = vect_feat.reshape(BATCH_N * MAX_N, F_N)
            input_mask_flat = input_mask.reshape(BATCH_N * MAX_N)
            vect_feat_out_flat = self.input_batchnorm(vect_feat_flat, input_mask_flat)
            vect_feat = vect_feat_out_flat.reshape(BATCH_N, MAX_N, F_N)

        G_features = self.gml(G, vect_feat, input_mask)
        if return_g_features:
            return G_features

        g_squeeze = G_features.squeeze(1)
        g_squeeze_flat = g_squeeze.reshape(-1, G_features.shape[-1])

        if self.resnet_out:
            x_1 = [m(g_squeeze_flat).reshape(BATCH_N, MAX_N, -1) for m in self.mix_out]
        else:
            x_1 = [m(g_squeeze) for m in self.mix_out]

        x_1 = torch.stack(x_1)

        if self.training:
            x_zeros = np.zeros(x_1.shape)
            if self.use_random_subsets:
                rand_ints = np.random.randint(x_1.shape[0], size=BATCH_N)
            else:
                rand_ints = (input_idx % len(self.mix_out)).cpu().numpy()
            # print(rand_ints)
            for i, v in enumerate(rand_ints):
                x_zeros[v, i, :, :] = 1
            x_1_sub = torch.Tensor(x_zeros).to(x_1.device) * x_1
            x_1_sub = x_1_sub.sum(dim=0)
        else:
            x_1_sub = x_1.mean(dim=0)
        # #print(x_1.shape)
        # idx = torch.randint(high=x_1.shape[0],
        #                     size=(BATCH_N, )).to(G.device)
        # #print("idx=", idx)
        # x_1_sub = torch.stack([x_1[v, v_i] for v_i, v in enumerate(idx)])
        std = torch.sqrt(torch.var(x_1, dim=0) + 1e-5)

        # print("numpy_std=", np.std(x_1.detach().cpu().numpy()))

        # kjlkjalds
        x_1 = x_1_sub

        return {"mu": x_1, "std": std}


class GraphMatLayerFastPow(nn.Module):
    def __init__(
        self,
        C,
        P,
        GS=1,
        mat_pow=1,
        mat_diag=False,
        noise=1e-6,
        agg_func=None,
        use_bias=False,
        nonlin=None,
        dropout=0.0,
        norm_by_neighbors=False,
    ):
        """ """
        super(GraphMatLayerFastPow, self).__init__()

        self.GS = GS
        self.noise = noise

        self.linlayers = nn.ModuleList()
        for ll in range(GS):
            l = nn.Linear(C, P, bias=use_bias)
            self.linlayers.append(l)
        self.dropout_rate = dropout

        if self.dropout_rate > 0:
            self.dropout_layers = nn.ModuleList(
                [nn.Dropout(self.dropout_rate) for _ in range(GS)]
            )

        # self.r = nn.PReLU()
        self.nonlin = nonlin
        if self.nonlin == "leakyrelu":
            self.r = nn.LeakyReLU()
        elif self.nonlin == "sigmoid":
            self.r = nn.Sigmoid()
        elif self.nonlin == "tanh":
            self.r = nn.Tanh()
        elif self.nonlin is None:
            pass
        else:
            raise ValueError(f"unknown nonlin {nonlin}")

        self.agg_func = agg_func
        self.mat_pow = mat_pow
        self.mat_diag = mat_diag

        self.norm_by_neighbors = norm_by_neighbors

    def forward(self, G, x):
        BATCH_N, CHAN_N, MAX_N, _ = G.shape

        def apply_ll(i, x):
            y = self.linlayers[i](x)
            if self.dropout_rate > 0.0:
                y = self.dropout_layers[i](y)
            return y

        Gprod = G
        for mp in range(self.mat_pow - 1):
            Gprod = torch.einsum("ijkl,ijlm->ijkm", G, Gprod)
        if self.mat_diag:
            Gprod = torch.eye(MAX_N).unsqueeze(0).unsqueeze(0).to(G.device) * Gprod

        multi_x = torch.stack([apply_ll(i, x) for i in range(self.GS)], 0)
        xout = torch.einsum("ijkl,jilm->jikm", [Gprod, multi_x])

        if self.norm_by_neighbors:
            G_neighbors = torch.clamp(G.sum(-1).permute(1, 0, 2), min=1)
            xout = xout / G_neighbors.unsqueeze(-1)

        if self.nonlin is not None:
            xout = self.r(xout)
        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)
        return xout


class GraphMatLayerFastPowSwap(nn.Module):
    def __init__(
        self,
        C,
        P,
        GS=1,
        mat_pow=1,
        mat_diag=False,
        noise=1e-6,
        agg_func=None,
        use_bias=False,
        nonlin=None,
        dropout=0.0,
        norm_by_neighbors=False,
    ):
        """ """
        super(GraphMatLayerFastPowSwap, self).__init__()

        self.GS = GS
        self.noise = noise

        self.linlayers = nn.ModuleList()
        for ll in range(GS):
            l = nn.Linear(C, P, bias=use_bias)
            self.linlayers.append(l)
        self.dropout_rate = dropout

        if self.dropout_rate > 0:
            self.dropout_layers = nn.ModuleList(
                [nn.Dropout(self.dropout_rate) for _ in range(GS)]
            )

        # self.r = nn.PReLU()
        self.nonlin = nonlin
        if self.nonlin == "leakyrelu":
            self.r = nn.LeakyReLU()
        elif self.nonlin is None:
            pass
        else:
            raise ValueError(f"unknown nonlin {nonlin}")

        self.agg_func = agg_func
        self.mat_pow = mat_pow
        self.mat_diag = mat_diag

        self.norm_by_neighbors = norm_by_neighbors

    def forward(self, G, x):
        BATCH_N, CHAN_N, MAX_N, _ = G.shape

        def apply_ll(i, x):
            y = self.linlayers[i](x)
            if self.dropout_rate > 0.0:
                y = self.dropout_layers[i](y)
            return y

        Gprod = G
        for mp in range(self.mat_pow - 1):
            Gprod = torch.einsum("ijkl,ijlm->ijkm", G, Gprod)
        if self.mat_diag:
            Gprod = torch.eye(MAX_N).unsqueeze(0).unsqueeze(0).to(G.device) * Gprod

        # multi_x = torch.stack([apply_ll(i,x) for i in range(self.GS)], 0)
        # xout = torch.einsum("ijkl,jilm->jikm", [Gprod, multi_x])
        # print("x.shape=", x.shape, "multi_x.shape=", multi_x.shape,
        #       "Gprod.shape=", Gprod.shape, "xout.shape=", xout.shape)

        x_adj = torch.einsum("ijkl,ilm->jikm", [Gprod, x])
        xout = torch.stack([apply_ll(i, x_adj[i]) for i in range(self.GS)])
        # print("\nx.shape=", x.shape,
        #       "x_adj.shape=", x_adj.shape,
        #       "Gprod.shape=", Gprod.shape,
        #       "xout.shape=", xout.shape)

        if self.norm_by_neighbors:
            G_neighbors = torch.clamp(G.sum(-1).permute(1, 0, 2), min=1)
            xout = xout / G_neighbors.unsqueeze(-1)

        if self.nonlin is not None:
            xout = self.r(xout)
        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)
        return xout


class GraphMatLayerFastPowSingleLayer(nn.Module):
    def __init__(
        self,
        C,
        P,
        GS=1,
        mat_pow=1,
        mat_diag=False,
        noise=1e-6,
        agg_func=None,
        use_bias=False,
        nonlin=None,
        dropout=0.0,
        norm_by_neighbors=False,
    ):
        """ """
        super(GraphMatLayerFastPowSingleLayer, self).__init__()

        self.GS = GS
        self.noise = noise

        self.l = nn.Linear(C, P, bias=use_bias)
        self.dropout_rate = dropout

        # if self.dropout_rate > 0:
        #     self.dropout_layers = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(GS)])

        # self.r = nn.PReLU()
        self.nonlin = nonlin
        if self.nonlin == "leakyrelu":
            self.r = nn.LeakyReLU()
        elif self.nonlin is None:
            pass
        else:
            raise ValueError(f"unknown nonlin {nonlin}")

        self.agg_func = agg_func
        self.mat_pow = mat_pow
        self.mat_diag = mat_diag

        self.norm_by_neighbors = norm_by_neighbors

    def forward(self, G, x):
        BATCH_N, CHAN_N, MAX_N, _ = G.shape

        def apply_ll(x):
            y = self.l(x)
            if self.dropout_rate > 0.0:
                y = self.dropout_layers(y)
            return y

        Gprod = G
        for mp in range(self.mat_pow - 1):
            Gprod = torch.einsum("ijkl,ijlm->ijkm", G, Gprod)
        if self.mat_diag:
            Gprod = torch.eye(MAX_N).unsqueeze(0).unsqueeze(0).to(G.device) * Gprod

        # multi_x = torch.stack([apply_ll(i,x) for i in range(self.GS)], 0)
        # xout = torch.einsum("ijkl,jilm->jikm", [Gprod, multi_x])
        # print("x.shape=", x.shape, "multi_x.shape=", multi_x.shape,
        #       "Gprod.shape=", Gprod.shape, "xout.shape=", xout.shape)

        x_adj = torch.einsum("ijkl,ilm->jikm", [Gprod, x])
        xout = torch.stack([apply_ll(x_adj[i]) for i in range(self.GS)])
        # print("\nx.shape=", x.shape,
        #       "x_adj.shape=", x_adj.shape,
        #       "Gprod.shape=", Gprod.shape,
        #       "xout.shape=", xout.shape)

        if self.norm_by_neighbors:
            G_neighbors = torch.clamp(G.sum(-1).permute(1, 0, 2), min=1)
            xout = xout / G_neighbors.unsqueeze(-1)

        if self.nonlin is not None:
            xout = self.r(xout)
        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)
        return xout


class GraphVertConfigBootstrap(nn.Module):
    def __init__(
        self,
        g_feature_n,
        g_feature_out_n=None,
        int_d=None,
        layer_n=None,
        mixture_n=5,
        resnet=True,
        gml_class="GraphMatLayers",
        gml_config={},
        init_noise=1e-5,
        init_bias=0.0,
        agg_func=None,
        GS=1,
        OUT_DIM=1,
        input_norm="batch",
        out_std=False,
        resnet_out=False,
        resnet_blocks=(3,),
        resnet_d=128,
        resnet_norm="layer",
        resnet_dropout=0.0,
        inner_norm=None,
        out_std_exp=False,
        force_lin_init=False,
        use_random_subsets=True,
    ):
        """ """
        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n
        print("g_feature_out_n=", g_feature_out_n)

        super(GraphVertConfigBootstrap, self).__init__()
        self.gml = eval(gml_class)(
            g_feature_n,
            g_feature_out_n,
            resnet=resnet,
            noise=init_noise,
            agg_func=parse_agg_func(agg_func),
            norm=inner_norm,
            GS=GS,
            **gml_config,
        )

        if input_norm == "batch":
            self.input_norm = MaskedBatchNorm1d(g_feature_n)
        elif input_norm == "layer":
            self.input_norm = MaskedLayerNorm1d(g_feature_n)
        else:
            self.input_norm = None

        self.resnet_out = resnet_out
        if not resnet_out:
            self.mix_out = nn.ModuleList(
                [nn.Linear(g_feature_out_n[-1], OUT_DIM) for _ in range(mixture_n)]
            )
        else:
            self.mix_out = nn.ModuleList(
                [
                    ResNetRegressionMaskedBN(
                        g_feature_out_n[-1],
                        block_sizes=resnet_blocks,
                        INT_D=resnet_d,
                        FINAL_D=resnet_d,
                        norm=resnet_norm,
                        dropout=resnet_dropout,
                        OUT_DIM=OUT_DIM,
                    )
                    for _ in range(mixture_n)
                ]
            )

        self.out_std = out_std
        self.out_std_exp = False

        self.use_random_subsets = use_random_subsets

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        if init_bias > 0:
                            nn.init.normal_(m.bias, 0, init_bias)
                        else:
                            nn.init.constant_(m.bias, 0)

    def forward(
        self,
        adj,
        vect_feat,
        input_mask,
        input_idx,
        adj_oh,
        return_g_features=False,
        also_return_g_features=False,
        **kwargs,
    ):
        G = adj

        BATCH_N, MAX_N, F_N = vect_feat.shape

        if self.input_norm is not None:
            vect_feat = apply_masked_1d_norm(self.input_norm, vect_feat, input_mask)

        G_features = self.gml(G, vect_feat, input_mask)
        if return_g_features:
            return G_features

        g_squeeze = G_features.squeeze(1)
        g_squeeze_flat = g_squeeze.reshape(-1, G_features.shape[-1])

        if self.resnet_out:
            x_1 = [
                m(g_squeeze_flat, input_mask.reshape(-1)).reshape(BATCH_N, MAX_N, -1)
                for m in self.mix_out
            ]
        else:
            x_1 = [m(g_squeeze) for m in self.mix_out]

        x_1 = torch.stack(x_1)

        if self.training:
            x_zeros = np.zeros(x_1.shape)
            if self.use_random_subsets:
                rand_ints = np.random.randint(x_1.shape[0], size=BATCH_N)
            else:
                rand_ints = (input_idx % len(self.mix_out)).cpu().numpy()
            # print(rand_ints)
            for i, v in enumerate(rand_ints):
                x_zeros[v, i, :, :] = 1
            x_1_sub = torch.Tensor(x_zeros).to(x_1.device) * x_1
            x_1_sub = x_1_sub.sum(dim=0)
        else:
            x_1_sub = x_1.mean(dim=0)
        # #print(x_1.shape)
        # idx = torch.randint(high=x_1.shape[0],
        #                     size=(BATCH_N, )).to(G.device)
        # #print("idx=", idx)
        # x_1_sub = torch.stack([x_1[v, v_i] for v_i, v in enumerate(idx)])
        if len(self.mix_out) > 1:
            std = torch.sqrt(torch.var(x_1, dim=0) + 1e-5)
        else:
            std = torch.ones_like(x_1_sub)

        # print("numpy_std=", np.std(x_1.detach().cpu().numpy()))

        # kjlkjalds
        x_1 = x_1_sub

        ret = {"mu": x_1, "std": std}
        if also_return_g_features:
            ret["g_features"] = g_squeeze
        return ret


class GraphMatLayerExpression(nn.Module):
    def __init__(
        self,
        C,
        P,
        GS=1,
        terms=[{"power": 1, "diag": False}],
        noise=1e-6,
        agg_func=None,
        use_bias=False,
        nonlin="leakyrelu",
        per_nonlin=None,
        dropout=0.0,
        cross_term_agg_func="sum",
        norm_by_neighbors=False,
    ):
        """
        Terms: [{'power': 3, 'diag': False}]

        """

        super(GraphMatLayerExpression, self).__init__()

        self.pow_ops = nn.ModuleList()
        for t in terms:
            l = GraphMatLayerFastPow(
                C,
                P,
                GS,
                mat_pow=t.get("power", 1),
                mat_diag=t.get("diag", False),
                noise=noise,
                use_bias=use_bias,
                nonlin=t.get("nonlin", per_nonlin),
                norm_by_neighbors=norm_by_neighbors,
                dropout=dropout,
            )
            self.pow_ops.append(l)

        self.nonlin = nonlin
        if self.nonlin == "leakyrelu":
            self.r = nn.LeakyReLU()
        elif self.nonlin == "relu":
            self.r = nn.ReLU()
        elif self.nonlin == "sigmoid":
            self.r = nn.Sigmoid()
        elif self.nonlin == "tanh":
            self.r = nn.Tanh()

        self.agg_func = agg_func
        self.cross_term_agg_func = cross_term_agg_func
        self.norm_by_neighbors = norm_by_neighbors

    def forward(self, G, x):
        BATCH_N, CHAN_N, MAX_N, _ = G.shape

        terms_stack = torch.stack([l(G, x) for l in self.pow_ops], dim=-1)

        if self.cross_term_agg_func == "sum":
            xout = torch.sum(terms_stack, dim=-1)
        elif self.cross_term_agg_func == "max":
            xout = torch.max(terms_stack, dim=-1)[0]
        elif self.cross_term_agg_func == "prod":
            xout = torch.prod(terms_stack, dim=-1)
        else:
            raise ValueError(f"unknown cross term agg func {self.cross_term_agg_func}")

        if self.nonlin is not None:
            xout = self.r(xout)
        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)
        return xout


def apply_masked_1d_norm(norm, x, mask):
    """
    Apply one of these norms and do the reshaping
    """
    F_N = x.shape[-1]
    x_flat = x.reshape(-1, F_N)
    mask_flat = mask.reshape(-1)
    out_flat = norm(x_flat, mask_flat)
    out = out_flat.reshape(*x.shape)
    return out


class GraphWithUncertainty(nn.Module):
    def __init__(
        self,
        g_feature_n,
        g_feature_out_n=None,
        int_d=None,
        common_layer_n=None,
        split_layer_n=2,
        resnet=True,
        gml_class="GraphMatLayers",
        gml_config={},
        init_noise=1e-5,
        init_bias=0.0,
        agg_func=None,
        GS=1,
        OUT_DIM=1,
        input_norm="batch",
        out_std=False,
        resnet_out=False,
        resnet_blocks=(3,),
        resnet_d=128,
        inner_norm=None,
        out_std_exp=False,
        force_lin_init=False,
        output_scale=1.0,
        output_std_scale=1.0,
        var_func="relu",
        use_random_subsets=True,
    ):
        """ """
        g_feature_out_n = [int_d] * common_layer_n

        super(GraphWithUncertainty, self).__init__()
        self.gml_shared = eval(gml_class)(
            g_feature_n,
            g_feature_out_n,
            resnet=resnet,
            noise=init_noise,
            agg_func=parse_agg_func(agg_func),
            norm=inner_norm,
            GS=GS,
            **gml_config,
        )

        split_f = [int_d] * split_layer_n
        self.gml_mean = eval(gml_class)(
            g_feature_out_n[-1],
            split_f,
            resnet=resnet,
            noise=init_noise,
            agg_func=parse_agg_func(agg_func),
            norm=inner_norm,
            GS=GS,
            **gml_config,
        )
        self.gml_var = eval(gml_class)(
            g_feature_out_n[-1],
            split_f,
            resnet=resnet,
            noise=init_noise,
            agg_func=parse_agg_func(agg_func),
            norm=inner_norm,
            GS=GS,
            **gml_config,
        )

        if input_norm == "batch":
            self.input_norm = MaskedBatchNorm1d(g_feature_n)
        elif input_norm == "layer":
            self.input_norm = MaskedLayerNorm1d(g_feature_n)
        else:
            self.input_norm = None

        self.res_mean = ResNetRegressionMaskedBN(
            split_f[-1],
            block_sizes=resnet_blocks,
            INT_D=resnet_d,
            FINAL_D=resnet_d,
            OUT_DIM=OUT_DIM,
        )
        self.res_var = ResNetRegressionMaskedBN(
            split_f[-1],
            block_sizes=resnet_blocks,
            INT_D=resnet_d,
            FINAL_D=resnet_d,
            OUT_DIM=OUT_DIM,
        )

        self.out_std = out_std
        self.out_std_exp = False
        self.output_scale = output_scale
        self.output_std_scale = output_std_scale

        self.use_random_subsets = use_random_subsets

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        if init_bias > 0:
                            nn.init.normal_(m.bias, 0, init_bias)
                        else:
                            nn.init.constant_(m.bias, 0)

        self.var_func = var_func

    def forward(
        self,
        adj,
        vect_feat,
        input_mask,
        input_idx,
        adj_oh,
        return_g_features=False,
        **kwargs,
    ):
        G = adj

        BATCH_N, MAX_N, F_N = vect_feat.shape

        if self.input_norm is not None:
            vect_feat = apply_masked_1d_norm(self.input_norm, vect_feat, input_mask)

        G_shared_features = self.gml_shared(G, vect_feat, input_mask)
        G_mean_features = self.gml_mean(G, G_shared_features, input_mask)
        G_var_features = self.gml_var(G, G_shared_features, input_mask)

        # if return_g_features:
        #     return G_features

        g_mean_squeeze_flat = G_mean_features.squeeze(1).reshape(
            -1, G_mean_features.shape[-1]
        )
        g_var_squeeze_flat = G_var_features.squeeze(1).reshape(
            -1, G_var_features.shape[-1]
        )

        x_mean = self.res_mean(g_mean_squeeze_flat, input_mask.reshape(-1)).reshape(
            BATCH_N, MAX_N, -1
        )
        x_var = self.res_var(
            g_var_squeeze_flat * self.output_std_scale, input_mask.reshape(-1)
        ).reshape(BATCH_N, MAX_N, -1)
        if self.var_func == "relu":
            x_var = F.ReLU(x_var)
        elif self.var_func == "softplus":
            x_var = F.softplus(x_var)
        elif self.var_func == "sigmoid":
            x_var = F.sigmoid(x_var)
        elif self.var_func == "exp":
            x_var = torch.exp(x_var)
        return {
            "mu": x_mean * self.output_scale,
            "std": x_var * self.output_scale,
        }


class GraphMatPerBondType(nn.Module):
    def __init__(
        self,
        input_feature_n,
        output_features_n,
        resnet=False,
        GS=1,
        norm=None,
        force_use_bias=False,
        noise=1e-5,
        agg_func=None,
        layer_class="GraphMatLayerFast",
        layer_config={},
    ):
        super(GraphMatPerBondType, self).__init__()

        self.gl = nn.ModuleList()
        self.resnet = resnet
        self.GS = GS
        self.agg_func = agg_func

        LayerClass = eval(layer_class)
        for li in range(len(output_features_n)):
            per_chan_l = nn.ModuleList()
            for c_i in range(GS):
                if li == 0:
                    gl = LayerClass(
                        input_feature_n,
                        output_features_n[0],
                        noise=noise,
                        agg_func=None,
                        GS=1,
                        use_bias=not norm or force_use_bias,
                        **layer_config,
                    )
                else:
                    gl = LayerClass(
                        output_features_n[li - 1],
                        output_features_n[li],
                        noise=noise,
                        agg_func=None,
                        GS=1,
                        use_bias=not norm or force_use_bias,
                        **layer_config,
                    )
                per_chan_l.append(gl)
            self.gl.append(per_chan_l)

        self.norm = norm
        if self.norm is not None:
            if self.norm == "batch":
                Nlayer = MaskedBatchNorm1d
            elif self.norm == "layer":
                Nlayer = MaskedLayerNorm1d

            self.bn = nn.ModuleList(
                [
                    nn.ModuleList([Nlayer(f) for _ in range(GS)])
                    for f in output_features_n
                ]
            )

    def forward(self, G, x, input_mask=None):
        x_per_chan = [x] * self.GS
        for gi, gl in enumerate(self.gl):
            for c_i in range(self.GS):
                x2 = gl[c_i](G[:, c_i : c_i + 1], x_per_chan[c_i]).squeeze()
                if self.norm:
                    x2 = self.bn[gi][c_i](
                        x2.reshape(-1, x2.shape[-1]), input_mask.reshape(-1)
                    ).reshape(x2.shape)

                if self.resnet and gi > 0:
                    x_per_chan[c_i] = x_per_chan[c_i] + x2
                else:
                    x_per_chan[c_i] = x2

        x_agg = torch.stack(x_per_chan, 1)
        x_out = self.agg_func(x_agg, 1)

        return x_out


class GraphMatPerBondTypeDebug(nn.Module):
    def __init__(
        self,
        input_feature_n,
        output_features_n,
        resnet=False,
        GS=1,
        norm=None,
        force_use_bias=False,
        noise=1e-5,
        agg_func=None,
        layer_class="GraphMatLayerFast",
        layer_config={},
    ):
        super(GraphMatPerBondTypeDebug, self).__init__()

        self.gl = nn.ModuleList()
        self.resnet = resnet
        self.GS = GS
        self.agg_func = agg_func

        LayerClass = eval(layer_class)
        for li in range(len(output_features_n)):
            per_chan_l = nn.ModuleList()
            for c_i in range(GS):
                if li == 0:
                    gl = LayerClass(
                        input_feature_n,
                        output_features_n[0],
                        noise=noise,
                        agg_func=None,
                        GS=1,
                        use_bias=not norm or force_use_bias,
                        **layer_config,
                    )
                else:
                    gl = LayerClass(
                        output_features_n[li - 1],
                        output_features_n[li],
                        noise=noise,
                        agg_func=None,
                        GS=1,
                        use_bias=not norm or force_use_bias,
                        **layer_config,
                    )
                per_chan_l.append(gl)
            self.gl.append(per_chan_l)

        self.norm = norm
        if self.norm is not None:
            if self.norm == "batch":
                Nlayer = MaskedBatchNorm1d
            elif self.norm == "layer":
                Nlayer = MaskedLayerNorm1d

            self.bn = nn.ModuleList(
                [
                    nn.ModuleList([Nlayer(f) for _ in range(GS)])
                    for f in output_features_n
                ]
            )

        self.final_l = nn.Linear(GS * output_features_n[-1], output_features_n[-1])

    def forward(self, G, x, input_mask=None):
        x_per_chan = [x] * self.GS
        for gi, gl in enumerate(self.gl):
            for c_i in range(self.GS):
                x2 = gl[c_i](G[:, c_i : c_i + 1], x_per_chan[c_i]).squeeze()
                if self.norm:
                    x2 = self.bn[gi][c_i](
                        x2.reshape(-1, x2.shape[-1]), input_mask.reshape(-1)
                    ).reshape(x2.shape)

                if self.resnet and gi > 0:
                    x_per_chan[c_i] = x_per_chan[c_i] + x2
                else:
                    x_per_chan[c_i] = x2

        x_agg = torch.cat(x_per_chan, -1)

        x_out = F.relu(self.final_l(x_agg))

        return x_out


class GraphMatPerBondTypeDebug2(nn.Module):
    def __init__(
        self,
        input_feature_n,
        output_features_n,
        resnet=False,
        GS=1,
        norm=None,
        force_use_bias=False,
        noise=1e-5,
        agg_func=None,
        layer_class="GraphMatLayerFast",
        layer_config={},
    ):
        super(GraphMatPerBondTypeDebug2, self).__init__()

        self.gl = nn.ModuleList()
        self.resnet = resnet
        self.GS = GS
        self.agg_func = agg_func

        LayerClass = eval(layer_class)

        self.cross_chan_lin = nn.ModuleList()
        for li in range(len(output_features_n)):
            per_chan_l = nn.ModuleList()
            for c_i in range(GS):
                if li == 0:
                    gl = LayerClass(
                        input_feature_n,
                        output_features_n[0],
                        noise=noise,
                        agg_func=None,
                        GS=1,
                        use_bias=not norm or force_use_bias,
                        **layer_config,
                    )
                else:
                    gl = LayerClass(
                        output_features_n[li - 1],
                        output_features_n[li],
                        noise=noise,
                        agg_func=None,
                        GS=1,
                        use_bias=not norm or force_use_bias,
                        **layer_config,
                    )
                per_chan_l.append(gl)
            self.gl.append(per_chan_l)
            self.cross_chan_lin.append(
                nn.Linear(GS * output_features_n[li], output_features_n[li])
            )

        self.norm = norm
        if self.norm is not None:
            if self.norm == "batch":
                Nlayer = MaskedBatchNorm1d
            elif self.norm == "layer":
                Nlayer = MaskedLayerNorm1d

            self.bn = nn.ModuleList(
                [
                    nn.ModuleList([Nlayer(f) for _ in range(GS)])
                    for f in output_features_n
                ]
            )

        self.final_l = nn.Linear(GS * output_features_n[-1], output_features_n[-1])

    def forward(self, G, x, input_mask=None):
        x_per_chan = [x] * self.GS
        for gi, gl in enumerate(self.gl):
            x_per_chan_latest = []
            for c_i in range(self.GS):
                x2 = gl[c_i](G[:, c_i : c_i + 1], x_per_chan[c_i]).squeeze()
                if self.norm:
                    x2 = self.bn[gi][c_i](
                        x2.reshape(-1, x2.shape[-1]), input_mask.reshape(-1)
                    ).reshape(x2.shape)
                x_per_chan_latest.append(x2)

            x_agg = torch.cat(x_per_chan_latest, -1)

            weight = self.cross_chan_lin[gi](x_agg)
            for c_i in range(self.GS):
                if self.resnet and gi > 0:
                    x_per_chan[c_i] = x_per_chan[c_i] + x_per_chan_latest[
                        c_i
                    ] * torch.sigmoid(weight)
                else:
                    x_per_chan[c_i] = x2

        x_agg = torch.cat(x_per_chan, -1)

        x_out = F.relu(self.final_l(x_agg))

        return x_out


class GraphVertConfig(nn.Module):
    def __init__(
        self,
        g_feature_n,
        g_feature_out_n=None,
        int_d=None,
        layer_n=None,
        mixture_n=5,
        resnet=True,
        gml_class="GraphMatLayers",
        gml_config={},
        init_noise=1e-5,
        init_bias=0.0,
        agg_func=None,
        GS=1,
        OUT_DIM=1,
        input_norm="batch",
        out_std=False,
        resnet_out=False,
        resnet_blocks=(3,),
        resnet_d=128,
        resnet_norm="layer",
        resnet_dropout=0.0,
        inner_norm=None,
        out_std_exp=False,
        force_lin_init=False,
        use_random_subsets=True,
    ):
        """ """
        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n
        print("g_feature_out_n=", g_feature_out_n)

        super(GraphVertConfig, self).__init__()
        self.gml = eval(gml_class)(
            g_feature_n,
            g_feature_out_n,
            resnet=resnet,
            noise=init_noise,
            agg_func=parse_agg_func(agg_func),
            norm=inner_norm,
            GS=GS,
            **gml_config,
        )

        if input_norm == "batch":
            self.input_norm = MaskedBatchNorm1d(g_feature_n)
        elif input_norm == "layer":
            self.input_norm = MaskedLayerNorm1d(g_feature_n)
        else:
            self.input_norm = None

        self.resnet_out = resnet_out

        if not resnet_out:
            self.lin_out = nn.Linear(g_feature_out_n[-1], OUT_DIM)
        else:
            self.lin_out = ResNetRegressionMaskedBN(
                g_feature_out_n[-1],
                block_sizes=resnet_blocks,
                INT_D=resnet_d,
                FINAL_D=resnet_d,
                norm=resnet_norm,
                dropout=resnet_dropout,
                OUT_DIM=OUT_DIM,
            )

        self.out_std = out_std
        self.out_std_exp = False

        self.use_random_subsets = use_random_subsets

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        if init_bias > 0:
                            nn.init.normal_(m.bias, 0, init_bias)
                        else:
                            nn.init.constant_(m.bias, 0)

    def forward(
        self,
        adj,
        vect_feat,
        input_mask,
        input_idx,
        adj_oh,
        return_g_features=False,
        also_return_g_features=False,
        **kwargs,
    ):
        G = adj

        BATCH_N, MAX_N, F_N = vect_feat.shape

        if self.input_norm is not None:
            vect_feat = apply_masked_1d_norm(self.input_norm, vect_feat, input_mask)

        G_features = self.gml(G, vect_feat, input_mask)
        if return_g_features:
            return G_features

        g_squeeze = G_features.squeeze(1)
        g_squeeze_flat = g_squeeze.reshape(-1, G_features.shape[-1])

        if self.resnet_out:
            x_1 = self.lin_out(g_squeeze_flat, input_mask.reshape(-1)).reshape(
                BATCH_N, MAX_N, -1
            )
        else:
            x_1 = self.lin_out(g_squeeze)

        ret = {"mu": x_1, "std": torch.ones_like(x_1)}
        if also_return_g_features:
            ret["g_features"] = g_squeeze
        return ret


def bootstrap_compute(x_1, input_idx, var_eps=1e-5, training=True):
    """
    shape is MIX_N, BATCH_SIZE, ....
    """
    MIX_N = x_1.shape[0]
    BATCH_N = x_1.shape[1]

    if training:
        x_zeros = np.zeros(x_1.shape)
        rand_ints = (input_idx % MIX_N).cpu().numpy()
        # print(rand_ints)
        for i, v in enumerate(rand_ints):
            x_zeros[v, i, :, :] = 1
        x_1_sub = torch.Tensor(x_zeros).to(x_1.device) * x_1
        x_1_sub = x_1_sub.sum(dim=0)
    else:
        x_1_sub = x_1.mean(dim=0)
    # x_1_sub = torch.stack([x_1[v, v_i] for v_i, v in enumerate(idx)])
    if MIX_N > 1:
        std = torch.sqrt(torch.var(x_1, dim=0) + var_eps)
    else:
        std = torch.ones_like(x_1_sub) * var_eps
    return x_1_sub, std


def bootstrap_perm_compute(x_1, input_idx, num_obs=1, var_eps=1e-5, training=True):
    """
    shape is MIX_N, BATCH_SIZE, ....
    compute bootstrap by taking the first num_obs instances of a permutation
    """
    MIX_N = x_1.shape[0]
    BATCH_N = x_1.shape[1]

    if training:
        x_zeros = np.zeros(x_1.shape)
        for i, idx in enumerate(input_idx):
            rs = np.random.RandomState(idx).permutation(MIX_N)[:num_obs]
            for j in range(num_obs):
                x_zeros[rs[j], i, :, :] = 1
        mask = torch.Tensor(x_zeros).to(x_1.device)
        x_1_sub = mask * x_1
        x_1_sub = x_1_sub.sum(dim=0) / num_obs
    else:
        x_1_sub = x_1.mean(dim=0)
    # x_1_sub = torch.stack([x_1[v, v_i] for v_i, v in enumerate(idx)])
    if MIX_N > 1:
        std = torch.sqrt(torch.var(x_1, dim=0) + var_eps)
    else:
        std = torch.ones_like(x_1_sub) * var_eps
    return x_1_sub, std


class PermMinLoss(nn.Module):
    """ """

    def __init__(self, norm="l2", scale=1.0, **kwargs):
        super(PermMinLoss, self).__init__()
        if norm == "l2":
            self.loss = nn.MSELoss()
        elif norm == "huber":
            self.loss = nn.SmoothL1Loss()

        self.scale = scale

    def __call__(self, pred, y, mask, vert_mask):
        mu = pred["mu"]
        assert mu.shape[2] == 1
        mu = mu.squeeze(-1)

        # pickle.dump({'mu' : mu.cpu().detach(),
        #              'y' : y.squeeze(-1).cpu().detach(),
        #              'mask' : mask.squeeze(-1).cpu().detach()},
        #             open("/tmp/test.debug", 'wb'))
        y_sorted, mask_sorted = util.min_assign(
            mu.cpu().detach(),
            y.squeeze(-1).cpu().detach(),
            mask.squeeze(-1).cpu().detach(),
        )
        y_sorted = y_sorted.to(y.device)
        mask_sorted = mask_sorted.to(mask.device)
        assert torch.sum(mask) > 0
        assert torch.sum(mask_sorted) > 0
        y_masked = y_sorted[mask_sorted > 0].reshape(-1, 1) * self.scale
        mu_masked = mu[mask_sorted > 0].reshape(-1, 1) * self.scale
        # print()
        # print("y_masked=", y_masked[:10].cpu().detach().numpy().flatten())
        # print("mu_masked=", mu_masked[:10].cpu().detach().numpy().flatten())

        l = self.loss(y_masked, mu_masked)
        if torch.isnan(l).any():
            print("loss is ", l, y_masked, mu_masked)

        return l


class GraphVertConfigBootstrapWithMultiMax(nn.Module):
    def __init__(
        self,
        g_feature_n,
        g_feature_out_n=None,
        int_d=None,
        layer_n=None,
        mixture_n=5,
        mixture_num_obs_per=1,
        resnet=True,
        gml_class="GraphMatLayers",
        gml_config={},
        init_noise=1e-5,
        init_bias=0.0,
        agg_func=None,
        GS=1,
        OUT_DIM=1,
        input_norm="batch",
        out_std=False,
        resnet_out=False,
        resnet_blocks=(3,),
        resnet_d=128,
        resnet_norm="layer",
        resnet_dropout=0.0,
        inner_norm=None,
        out_std_exp=False,
        force_lin_init=False,
        use_random_subsets=True,
    ):
        """
        GraphVertConfigBootstrap with multiple max outs
        """
        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n
        print("g_feature_out_n=", g_feature_out_n)

        super(GraphVertConfigBootstrapWithMultiMax, self).__init__()
        self.gml = eval(gml_class)(
            g_feature_n,
            g_feature_out_n,
            resnet=resnet,
            noise=init_noise,
            agg_func=parse_agg_func(agg_func),
            norm=inner_norm,
            GS=GS,
            **gml_config,
        )

        if input_norm == "batch":
            self.input_norm = MaskedBatchNorm1d(g_feature_n)
        elif input_norm == "layer":
            self.input_norm = MaskedLayerNorm1d(g_feature_n)
        else:
            self.input_norm = None

        self.resnet_out = resnet_out
        if not resnet_out:
            self.mix_out = nn.ModuleList(
                [nn.Linear(g_feature_out_n[-1], OUT_DIM) for _ in range(mixture_n)]
            )
        else:
            self.mix_out = nn.ModuleList(
                [
                    ResNetRegressionMaskedBN(
                        g_feature_out_n[-1],
                        block_sizes=resnet_blocks,
                        INT_D=resnet_d,
                        FINAL_D=resnet_d,
                        norm=resnet_norm,
                        dropout=resnet_dropout,
                        OUT_DIM=OUT_DIM,
                    )
                    for _ in range(mixture_n)
                ]
            )

        self.out_std = out_std
        self.out_std_exp = False

        self.use_random_subsets = use_random_subsets
        self.mixture_num_obs_per = mixture_num_obs_per

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        if init_bias > 0:
                            nn.init.normal_(m.bias, 0, init_bias)
                        else:
                            nn.init.constant_(m.bias, 0)

    def forward(
        self,
        adj,
        vect_feat,
        input_mask,
        input_idx,
        adj_oh,
        return_g_features=False,
        also_return_g_features=False,
        **kwargs,
    ):
        G = adj

        BATCH_N, MAX_N, F_N = vect_feat.shape

        if self.input_norm is not None:
            vect_feat = apply_masked_1d_norm(self.input_norm, vect_feat, input_mask)

        G_features = self.gml(G, vect_feat, input_mask)
        if return_g_features:
            return G_features

        g_squeeze = G_features.squeeze(1)
        g_squeeze_flat = g_squeeze.reshape(-1, G_features.shape[-1])

        if self.resnet_out:
            x_1 = [
                m(g_squeeze_flat, input_mask.reshape(-1)).reshape(BATCH_N, MAX_N, -1)
                for m in self.mix_out
            ]
        else:
            x_1 = [m(g_squeeze) for m in self.mix_out]

        x_1 = torch.stack(x_1)

        x_1, std = bootstrap_perm_compute(
            x_1, input_idx, self.mixture_num_obs_per, training=self.training
        )

        ret = {"shift_mu": x_1, "shift_std": std}
        if also_return_g_features:
            ret["g_features"] = g_squeeze
        return ret


class GraphMatLayerExpressionWNorm(nn.Module):
    def __init__(
        self,
        C,
        P,
        GS=1,
        terms=[{"power": 1, "diag": False}],
        noise=1e-6,
        agg_func=None,
        use_bias=False,
        post_agg_nonlin=None,
        post_agg_norm=None,
        per_nonlin=None,
        dropout=0.0,
        cross_term_agg_func="sum",
        norm_by_neighbors=False,
    ):
        """ """

        super(GraphMatLayerExpressionWNorm, self).__init__()

        self.pow_ops = nn.ModuleList()
        for t in terms:
            l = GraphMatLayerFastPow(
                C,
                P,
                GS,
                mat_pow=t.get("power", 1),
                mat_diag=t.get("diag", False),
                noise=noise,
                use_bias=use_bias,
                nonlin=t.get("nonlin", per_nonlin),
                norm_by_neighbors=norm_by_neighbors,
                dropout=dropout,
            )
            self.pow_ops.append(l)

        self.post_agg_nonlin = post_agg_nonlin
        if self.post_agg_nonlin == "leakyrelu":
            self.r = nn.LeakyReLU()
        elif self.post_agg_nonlin == "relu":
            self.r = nn.ReLU()
        elif self.post_agg_nonlin == "sigmoid":
            self.r = nn.Sigmoid()
        elif self.post_agg_nonlin == "tanh":
            self.r = nn.Tanh()

        self.agg_func = agg_func
        self.cross_term_agg_func = cross_term_agg_func
        self.norm_by_neighbors = norm_by_neighbors
        self.post_agg_norm = post_agg_norm
        if post_agg_norm == "layer":
            self.pa_norm = nn.LayerNorm(P)

        elif post_agg_norm == "batch":
            self.pa_norm = nn.BatchNorm1d(P)

    def forward(self, G, x):
        BATCH_N, CHAN_N, MAX_N, _ = G.shape

        terms_stack = torch.stack([l(G, x) for l in self.pow_ops], dim=-1)

        if self.cross_term_agg_func == "sum":
            xout = torch.sum(terms_stack, dim=-1)
        elif self.cross_term_agg_func == "max":
            xout = torch.max(terms_stack, dim=-1)[0]
        elif self.cross_term_agg_func == "prod":
            xout = torch.prod(terms_stack, dim=-1)
        else:
            raise ValueError(f"unknown cross term agg func {self.cross_term_agg_func}")

        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)

        if self.post_agg_nonlin is not None:
            xout = self.r(xout)
        if self.post_agg_norm is not None:
            xout = self.pa_norm(xout.reshape(-1, xout.shape[-1])).reshape(xout.shape)

        return xout


class GraphMatLayerFastPow2(nn.Module):
    def __init__(
        self,
        C,
        P,
        GS=1,
        mat_pow=1,
        mat_diag=False,
        noise=1e-6,
        agg_func=None,
        use_bias=False,
        nonlin=None,
        dropout=0.0,
        norm_by_neighbors=False,
    ):
        """
        Two layer MLP

        """
        super(GraphMatLayerFastPow2, self).__init__()

        self.GS = GS
        self.noise = noise

        self.linlayers1 = nn.ModuleList()
        self.linlayers2 = nn.ModuleList()

        for ll in range(GS):
            l = nn.Linear(C, P)
            self.linlayers1.append(l)
            l = nn.Linear(P, P)
            self.linlayers2.append(l)
        self.dropout_rate = dropout

        if self.dropout_rate > 0:
            self.dropout_layers = nn.ModuleList(
                [nn.Dropout(self.dropout_rate) for _ in range(GS)]
            )

        # self.r = nn.PReLU()
        self.nonlin = nonlin
        if self.nonlin == "leakyrelu":
            self.r = nn.LeakyReLU()
        elif self.nonlin == "sigmoid":
            self.r = nn.Sigmoid()
        elif self.nonlin == "tanh":
            self.r = nn.Tanh()
        elif self.nonlin is None:
            pass
        else:
            raise ValueError(f"unknown nonlin {nonlin}")

        self.agg_func = agg_func
        self.mat_pow = mat_pow
        self.mat_diag = mat_diag

        self.norm_by_neighbors = norm_by_neighbors

    def forward(self, G, x):
        BATCH_N, CHAN_N, MAX_N, _ = G.shape

        def apply_ll(i, x):
            y = F.relu(self.linlayers1[i](x))
            y = self.linlayers2[i](y)

            if self.dropout_rate > 0.0:
                y = self.dropout_layers[i](y)
            return y

        Gprod = G
        for mp in range(self.mat_pow - 1):
            Gprod = torch.einsum("ijkl,ijlm->ijkm", G, Gprod)
        if self.mat_diag:
            Gprod = torch.eye(MAX_N).unsqueeze(0).unsqueeze(0).to(G.device) * Gprod
        multi_x = torch.stack([apply_ll(i, x) for i in range(self.GS)], 0)
        # print("Gprod.shape=", Gprod.shape, "multi_x.shape=", multi_x.shape)
        xout = torch.einsum("ijkl,jilm->jikm", [Gprod, multi_x])

        if self.norm_by_neighbors != False:
            G_neighbors = torch.clamp(G.sum(-1).permute(1, 0, 2), min=1)
            if self.norm_by_neighbors == "sqrt":
                xout = xout / torch.sqrt(G_neighbors.unsqueeze(-1))

            else:
                xout = xout / G_neighbors.unsqueeze(-1)

        if self.nonlin is not None:
            xout = self.r(xout)
        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)
        return xout


class GraphMatLayerExpressionWNorm2(nn.Module):
    def __init__(
        self,
        C,
        P,
        GS=1,
        terms=[{"power": 1, "diag": False}],
        noise=1e-6,
        agg_func=None,
        use_bias=False,
        post_agg_nonlin=None,
        post_agg_norm=None,
        per_nonlin=None,
        dropout=0.0,
        cross_term_agg_func="sum",
        norm_by_neighbors=False,
    ):
        """ """

        super(GraphMatLayerExpressionWNorm2, self).__init__()

        self.pow_ops = nn.ModuleList()
        for t in terms:
            l = GraphMatLayerFastPow2(
                C,
                P,
                GS,
                mat_pow=t.get("power", 1),
                mat_diag=t.get("diag", False),
                noise=noise,
                use_bias=use_bias,
                nonlin=t.get("nonlin", per_nonlin),
                norm_by_neighbors=norm_by_neighbors,
                dropout=dropout,
            )
            self.pow_ops.append(l)

        self.post_agg_nonlin = post_agg_nonlin
        if self.post_agg_nonlin == "leakyrelu":
            self.r = nn.LeakyReLU()
        elif self.post_agg_nonlin == "relu":
            self.r = nn.ReLU()
        elif self.post_agg_nonlin == "sigmoid":
            self.r = nn.Sigmoid()
        elif self.post_agg_nonlin == "tanh":
            self.r = nn.Tanh()

        self.agg_func = agg_func
        self.cross_term_agg_func = cross_term_agg_func
        self.norm_by_neighbors = norm_by_neighbors
        self.post_agg_norm = post_agg_norm
        if post_agg_norm == "layer":
            self.pa_norm = nn.LayerNorm(P)

        elif post_agg_norm == "batch":
            self.pa_norm = nn.BatchNorm1d(P)

    def forward(self, G, x):
        BATCH_N, CHAN_N, MAX_N, _ = G.shape

        terms_stack = torch.stack([l(G, x) for l in self.pow_ops], dim=-1)

        if self.cross_term_agg_func == "sum":
            xout = torch.sum(terms_stack, dim=-1)
        elif self.cross_term_agg_func == "max":
            xout = torch.max(terms_stack, dim=-1)[0]
        elif self.cross_term_agg_func == "prod":
            xout = torch.prod(terms_stack, dim=-1)
        else:
            raise ValueError(f"unknown cross term agg func {self.cross_term_agg_func}")

        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)

        if self.post_agg_nonlin is not None:
            xout = self.r(xout)
        if self.post_agg_norm is not None:
            xout = self.pa_norm(xout.reshape(-1, xout.shape[-1])).reshape(xout.shape)

        return xout


class Ensemble(nn.Module):
    def __init__(
        self, g_feature_n, GS, ensemble_class, ensemble_n, ensemble_config={}, **kwargs
    ):
        """
        Combine a bunch of other nets

        """

        super(Ensemble, self).__init__()
        for k, v in ensemble_config.items():
            print(f"{k}={v}")

        self.ensemble = nn.ModuleList(
            [
                eval(ensemble_class)(g_feature_n=g_feature_n, GS=GS, **ensemble_config)
                for _ in range(ensemble_n)
            ]
        )

    def forward(self, *args, **kwargs):
        out = [l(*args, **kwargs) for l in self.ensemble]

        # how do we output our mean, std?

        mu = torch.mean(torch.stack([o["mu"] for o in out], dim=0), dim=0)
        std = torch.sqrt(
            torch.sum(torch.stack([o["std"] ** 2 for o in out], dim=0), dim=0)
        )

        return {"mu": mu, "std": std, "per_out": out}


class EnsembleLoss(nn.Module):
    """ """

    def __init__(self, subloss_name, **kwargs):
        super(EnsembleLoss, self).__init__()

        self.l = eval(subloss_name)(**kwargs)

    def __call__(self, pred, y, mask, vert_mask):
        agg_loss = [self.l(o, y, mask, vert_mask).reshape(1) for o in pred["per_out"]]

        return torch.mean(torch.cat(agg_loss))


def create_nonlin(nonlin):
    if nonlin == "leakyrelu":
        r = nn.LeakyReLU()
    elif nonlin == "sigmoid":
        r = nn.Sigmoid()
    elif nonlin == "tanh":
        r = nn.Tanh()
    elif nonlin == "relu":
        r = nn.ReLU()
    elif nonlin == "identity":
        r = nn.Identity()
    else:
        raise ValueError(f"unknown nonlin {nonlin}")

    return r


class GCNLDLayer(nn.Module):
    def __init__(
        self,
        C,
        P,
        GS=1,
        terms=[{"power": 1, "diag": False}],
        mlp_config={"layer_n": 1, "nonlin": "leakyrelu"},
        chanagg="pre",
        dropout=0.0,
        learn_w=True,
        norm_by_degree=False,
        **kwargs,
    ):
        """ """
        super(GCNLDLayer, self).__init__()
        self.terms = terms
        self.C = C
        self.P = P
        if learn_w:
            self.scalar_weights = nn.Parameter(torch.zeros(len(terms)))
        else:
            self.scalar_weights = torch.zeros(len(terms))

        self.chanagg = chanagg
        self.norm_by_degree = norm_by_degree

        if self.chanagg == "cat":
            self.out_lin = MLP(input_d=C * GS, output_d=P, d=P, **mlp_config)
        else:
            self.out_lin = MLP(input_d=C, output_d=P, d=P, **mlp_config)

        self.dropout_p = dropout
        if self.dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout)

    def mpow(self, G, k):
        Gprod = G
        for i in range(k - 1):
            Gprod = torch.einsum("ijkl,ijlm->ijkm", G, Gprod)
        return Gprod

    def forward(self, G, x):
        BATCH_N, CHAN_N, MAX_N, _ = G.shape

        # first compute each power
        Gdiag = torch.eye(MAX_N).unsqueeze(0).unsqueeze(0).to(G.device)

        G_terms = torch.zeros_like(G)
        for ti, t in enumerate(self.terms):
            G_pow = self.mpow(G, t["power"])
            if t.get("diag", False):
                G_pow = G_pow * Gdiag
            G_terms = G_terms + G_pow * torch.sigmoid(self.scalar_weights[ti])

        Xp = G_terms @ x.unsqueeze(1)

        # normalization
        G_norm = torch.clamp(G.sum(dim=-1), min=1)
        if self.norm_by_degree:
            Xp = Xp / G_norm.unsqueeze(-1)

        if self.chanagg == "cat":
            a = Xp.permute(0, 2, 3, 1)
            Xp = a.reshape(a.shape[0], a.shape[1], -1)
        X = self.out_lin(Xp)
        if self.dropout_p > 0:
            X = self.dropout(X)

        if self.chanagg == "goodmax":
            X = goodmax(X, 1)

        return X


class MLP(nn.Module):
    def __init__(
        self,
        layer_n=1,
        d=128,
        input_d=None,
        output_d=None,
        nonlin="relu",
        final_nonlin=True,
        use_bias=True,
    ):
        super(MLP, self).__init__()

        ml = []
        for i in range(layer_n):
            in_d = d
            out_d = d
            if i == 0 and input_d is not None:
                in_d = input_d
            if (i == (layer_n - 1)) and output_d is not None:
                out_d = output_d

            linlayer = nn.Linear(in_d, out_d, use_bias)

            ml.append(linlayer)
            nonlin_layer = create_nonlin(nonlin)
            if i == (layer_n - 1) and not final_nonlin:
                pass
            else:
                ml.append(nonlin_layer)
        self.ml = nn.Sequential(*ml)

    def forward(self, x):
        return self.ml(x)


class GCNLDLinPerChanLayer(nn.Module):
    def __init__(
        self,
        C,
        P,
        GS=1,
        terms=[{"power": 1, "diag": False}],
        nonlin="leakyrelu",
        chanagg="pre",
        dropout=0.0,
        learn_w=True,
        norm_by_degree="degree",
        w_transform="sigmoid",
        mlp_config={"layer_n": 1, "nonlin": "leakyrelu"},
        **kwargs,
    ):
        """ """
        super(GCNLDLinPerChanLayer, self).__init__()
        self.terms = terms
        self.C = C
        self.P = P
        if learn_w:
            self.scalar_weights = nn.Parameter(torch.zeros(len(terms)))
        else:
            self.scalar_weights = torch.zeros(len(terms))

        self.chanagg = chanagg

        self.out_lin = nn.ModuleList(
            [MLP(input_d=C, d=P, output_d=P, **mlp_config) for _ in range(GS)]
        )
        self.w_transform = w_transform
        self.dropout_p = dropout
        if self.dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout)

        self.norm_by_degree = norm_by_degree

    def mpow(self, G, k):
        Gprod = G
        for i in range(k - 1):
            Gprod = torch.einsum("ijkl,ijlm->ijkm", G, Gprod)
        return Gprod

    def forward(self, G, x):
        BATCH_N, CHAN_N, MAX_N, _ = G.shape

        # first compute each power
        Gdiag = torch.eye(MAX_N).unsqueeze(0).unsqueeze(0).to(G.device)

        G_terms = torch.zeros_like(G)
        for ti, t in enumerate(self.terms):
            if self.w_transform == "sigmoid":
                w = torch.sigmoid(self.scalar_weights[ti])
            elif self.w_transform == "tanh":
                w = torch.tanh(self.scalar_weights[ti])

            G_pow = self.mpow(G, t["power"])
            if t.get("diag", False):
                G_pow = G_pow * Gdiag
            G_terms = G_terms + G_pow * w

        # normalization
        if self.norm_by_degree == "degree":
            G_norm = torch.clamp(G.sum(dim=-1), min=1)
            G_terms = G_terms / G_norm.unsqueeze(-1)
        elif self.norm_by_degree == "total":
            G_norm = torch.clamp(G_terms.sum(dim=-1), min=1)
            G_terms = G_terms / G_norm.unsqueeze(-1)

        Xp = G_terms @ x.unsqueeze(1)

        XP0 = Xp.permute(1, 0, 2, 3)

        X = [l(x) for l, x in zip(self.out_lin, XP0)]
        X = torch.stack(X)
        if self.dropout_p > 0:
            X = self.dropout(X)

        if self.chanagg == "goodmax":
            X = goodmax(X, 0)

        return X


class GCNLDLinPerChanLayerDEBUG(nn.Module):
    def __init__(
        self,
        C,
        P,
        GS=1,
        terms=[{"power": 1, "diag": False}],
        nonlin="leakyrelu",
        chanagg="pre",
        dropout=0.0,
        learn_w=True,
        norm_by_degree="degree",
        w_transform="sigmoid",
        mlp_config={"layer_n": 1, "nonlin": "leakyrelu"},
        **kwargs,
    ):
        """ """
        super(GCNLDLinPerChanLayerDEBUG, self).__init__()
        self.terms = terms
        self.C = C
        self.P = P
        if learn_w:
            self.scalar_weights = nn.Parameter(torch.zeros(len(terms)))
        else:
            self.scalar_weights = torch.zeros(len(terms))

        self.chanagg = chanagg

        self.out_lin = nn.ModuleList(
            [MLP(input_d=C, d=P, output_d=P, **mlp_config) for _ in range(GS)]
        )
        self.w_transform = w_transform
        self.dropout_p = dropout
        if self.dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout)

        self.norm_by_degree = norm_by_degree

    def mpow(self, G, k):
        Gprod = G
        for i in range(k - 1):
            Gprod = torch.einsum("ijkl,ijlm->ijkm", G, Gprod)
        return Gprod

    def forward(self, G, x):
        BATCH_N, CHAN_N, MAX_N, _ = G.shape

        G_embed = self.chan_embed(G.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # first compute each power
        Gdiag = torch.eye(MAX_N).unsqueeze(0).unsqueeze(0).to(G.device)

        G_terms = torch.zeros_like(G)
        for ti, t in enumerate(self.terms):
            if self.w_transform == "sigmoid":
                w = torch.sigmoid(self.scalar_weights[ti])
            elif self.w_transform == "tanh":
                w = torch.tanh(self.scalar_weights[ti])

            G_pow = self.mpow(G, t["power"])
            if t.get("diag", False):
                G_pow = G_pow * Gdiag
            G_terms = G_terms + G_pow * w

        # normalization
        if self.norm_by_degree == "degree":
            G_norm = torch.clamp(G.sum(dim=-1), min=1)
            G_terms = G_terms / G_norm.unsqueeze(-1)
        elif self.norm_by_degree == "total":
            G_norm = torch.clamp(G_terms.sum(dim=-1), min=1)
            G_terms = G_terms / G_norm.unsqueeze(-1)

        X = [l(x) for l in self.out_lin]
        X = torch.stack(X, 1)
        # print("X.shape=", X.shape, "G_terms.shape=", G_terms.shape)
        # X = torch.clamp(G_terms, max=1) @ X
        X = G_terms @ X

        # Xp = G_terms @ x.unsqueeze(1)

        # XP0 = Xp.permute(1, 0, 2, 3)
        # X = [l(x) for l, x in zip(self.out_lin, XP0)]

        # print("Xout.shape=", X.shape)
        # lkhasdlsaj
        if self.dropout_p > 0:
            X = self.dropout(X)

        if self.chanagg == "goodmax":
            X = goodmax(X, 1)
        elif self.chanagg == "sum":
            X = torch.sum(X, 1)
        elif self.chanagg == "mean":
            X = torch.mean(X, 1)

        return X


class GCNLDLinPerChanLayerEdgeEmbed(nn.Module):
    def __init__(
        self,
        C,
        P,
        GS=1,
        terms=[{"power": 1, "diag": False}],
        nonlin="leakyrelu",
        chanagg="pre",
        dropout=0.0,
        learn_w=True,
        embed_dim_multiple=1,
        embed_transform=None,
        norm_by_degree="degree",
        w_transform="sigmoid",
        mlp_config={"layer_n": 1, "nonlin": "leakyrelu"},
        **kwargs,
    ):
        """ """
        super(GCNLDLinPerChanLayerEdgeEmbed, self).__init__()
        self.terms = terms
        self.C = C
        self.P = P
        if learn_w:
            self.scalar_weights = nn.Parameter(torch.zeros(len(terms)))
        else:
            self.scalar_weights = torch.zeros(len(terms))

        self.chanagg = chanagg

        self.chan_embed = nn.Linear(GS, GS * embed_dim_multiple)

        self.out_lin = nn.ModuleList(
            [
                MLP(input_d=C, d=P, output_d=P, **mlp_config)
                for _ in range(GS * embed_dim_multiple)
            ]
        )
        self.w_transform = w_transform
        self.dropout_p = dropout
        if self.dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout)

        self.norm_by_degree = norm_by_degree
        self.embed_transform = embed_transform

    def mpow(self, G, k):
        Gprod = G
        for i in range(k - 1):
            Gprod = torch.einsum("ijkl,ijlm->ijkm", G, Gprod)
        return Gprod

    def forward(self, G, x):
        BATCH_N, CHAN_N, MAX_N, _ = G.shape

        G_embed = self.chan_embed(G.permute(0, 2, 3, 1))
        if self.embed_transform == "sigmoid":
            G_embed = torch.sigmoid(G_embed)
        elif self.embed_transform == "softmax":
            G_embed = torch.softmax(G_embed, -1)

        G = G_embed.permute(0, 3, 1, 2)
        # first compute each power
        Gdiag = torch.eye(MAX_N).unsqueeze(0).unsqueeze(0).to(G.device)

        G_terms = torch.zeros_like(G)
        for ti, t in enumerate(self.terms):
            if self.w_transform == "sigmoid":
                w = torch.sigmoid(self.scalar_weights[ti])
            elif self.w_transform == "tanh":
                w = torch.tanh(self.scalar_weights[ti])

            G_pow = self.mpow(G, t["power"])
            if t.get("diag", False):
                G_pow = G_pow * Gdiag
            G_terms = G_terms + G_pow * w

        # normalization
        if self.norm_by_degree == "degree":
            G_norm = torch.clamp(G.sum(dim=-1), min=1)
            G_terms = G_terms / G_norm.unsqueeze(-1)
        elif self.norm_by_degree == "total":
            G_norm = torch.clamp(G_terms.sum(dim=-1), min=1)
            G_terms = G_terms / G_norm.unsqueeze(-1)

        X = [l(x) for l in self.out_lin]
        X = torch.stack(X, 1)

        if self.dropout_p > 0:
            X = self.dropout(X)

        # print("X.shape=", X.shape, "G_terms.shape=", G_terms.shape)
        # X = torch.clamp(G_terms, max=1) @ X
        X = G_terms @ X

        # Xp = G_terms @ x.unsqueeze(1)

        # XP0 = Xp.permute(1, 0, 2, 3)
        # X = [l(x) for l, x in zip(self.out_lin, XP0)]

        # print("Xout.shape=", X.shape)
        # lkhasdlsaj
        # if self.dropout_p > 0:
        #     X = self.dropout(X)

        if self.chanagg == "goodmax":
            X = goodmax(X, 1)
        elif self.chanagg == "sum":
            X = torch.sum(X, 1)
        elif self.chanagg == "mean":
            X = torch.mean(X, 1)

        return X


class GCNLDLinPerChanLayerAttn(nn.Module):
    def __init__(
        self,
        C,
        P,
        GS=1,
        terms=[{"power": 1, "diag": False}],
        nonlin="leakyrelu",
        chanagg="pre",
        dropout=0.0,
        # learn_w = True,
        norm_by_degree="degree",
        # w_transform = 'sigmoid',
        mlp_config={"layer_n": 1, "nonlin": "leakyrelu"},
        **kwargs,
    ):
        """ """
        super(GCNLDLinPerChanLayerAttn, self).__init__()
        self.terms = terms
        self.C = C
        self.P = P
        # if learn_w:
        #     self.scalar_weights = nn.Parameter(torch.zeros(len(terms)))
        # else:
        #     self.scalar_weights = torch.zeros(len(terms))

        self.chanagg = chanagg

        self.out_lin = nn.ModuleList(
            [MLP(input_d=C, d=P, output_d=P, **mlp_config) for _ in range(GS)]
        )
        # self.w_transform = w_transform
        self.dropout_p = dropout
        if self.dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout)

        self.norm_by_degree = norm_by_degree

        self.term_attn = MLP(
            input_d=self.C, d=128, layer_n=3, output_d=len(terms), final_nonlin=False
        )

    def mpow(self, G, k):
        Gprod = G
        for i in range(k - 1):
            Gprod = torch.einsum("ijkl,ijlm->ijkm", G, Gprod)
        return Gprod

    def forward(self, G, x):
        BATCH_N, CHAN_N, MAX_N, _ = G.shape

        # first compute each power
        Gdiag = torch.eye(MAX_N).unsqueeze(0).unsqueeze(0).to(G.device)

        G_terms = []
        for ti, t in enumerate(self.terms):
            # if self.w_transform == 'sigmoid':
            #     w = torch.sigmoid(self.scalar_weights[ti])
            # elif self.w_transform == 'tanh':
            #     w = torch.tanh(self.scalar_weights[ti])

            G_pow = self.mpow(G, t["power"])
            if t.get("diag", False):
                G_pow = G_pow * Gdiag
            G_terms.append(G_pow)

        # normalization
        if self.norm_by_degree == "degree":
            G_norm = torch.clamp(G.sum(dim=-1), min=1)
            G_terms = [G_term / G_norm.unsqueeze(-1) for G_term in G_terms]
        elif self.norm_by_degree == "total":
            G_norm = torch.clamp(G_terms.sum(dim=-1), min=1)
            G_terms = [G_term / G_norm.unsqueeze(-1) for G_term in G_terms]

        X = [l(x) for l in self.out_lin]
        X = torch.stack(X, 1)

        if self.dropout_p > 0:
            X = self.dropout(X)

        attention = torch.softmax(self.term_attn(x), -1)
        # attention = torch.sigmoid(self.term_attn(x))
        # print("X.shape=", X.shape, "G_terms.shape=", G_terms.shape)
        Xterms = torch.stack([G_term @ X for G_term in G_terms], -1)
        attention = attention.unsqueeze(1).unsqueeze(3)
        # print("Xterms.shape=", Xterms.shape,
        #      "attention.shape=", attention.shape)
        X = (Xterms * attention).sum(dim=-1)

        # Xp = G_terms @ x.unsqueeze(1)

        # XP0 = Xp.permute(1, 0, 2, 3)
        # X = [l(x) for l, x in zip(self.out_lin, XP0)]

        # print("Xout.shape=", X.shape)
        # lkhasdlsaj
        if self.chanagg == "goodmax":
            X = goodmax(X, 1)

        return X


class DropoutEmbedExp(nn.Module):
    def __init__(
        self,
        g_feature_n,
        g_feature_out_n=None,
        int_d=None,
        layer_n=None,
        mixture_n=5,
        mixture_num_obs_per=1,
        resnet=True,
        gml_class="GraphMatLayers",
        gml_config={},
        init_noise=1e-5,
        init_bias=0.0,
        agg_func=None,
        GS=1,
        OUT_DIM=1,
        input_norm="batch",
        out_std=False,
        resnet_out=False,
        resnet_blocks=(3,),
        resnet_d=128,
        resnet_norm="layer",
        resnet_dropout=0.0,
        inner_norm=None,
        out_std_exp=False,
        force_lin_init=False,
        use_random_subsets=True,
        input_vert_dropout_p=0.0,
        input_edge_dropout_p=0.0,
        embed_edges=False,
    ):
        """
        GraphVertConfigBootstrap with multiple max outs
        """
        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n
        print("g_feature_out_n=", g_feature_out_n)

        super(DropoutEmbedExp, self).__init__()
        self.gml = eval(gml_class)(
            g_feature_n,
            g_feature_out_n,
            resnet=resnet,
            noise=init_noise,
            agg_func=parse_agg_func(agg_func),
            norm=inner_norm,
            GS=GS,
            **gml_config,
        )

        if input_norm == "batch":
            self.input_norm = MaskedBatchNorm1d(g_feature_n)
        elif input_norm == "layer":
            self.input_norm = MaskedLayerNorm1d(g_feature_n)
        else:
            self.input_norm = None

        self.resnet_out = resnet_out
        if not resnet_out:
            self.mix_out = nn.ModuleList(
                [nn.Linear(g_feature_out_n[-1], OUT_DIM) for _ in range(mixture_n)]
            )
        else:
            self.mix_out = nn.ModuleList(
                [
                    ResNetRegressionMaskedBN(
                        g_feature_out_n[-1],
                        block_sizes=resnet_blocks,
                        INT_D=resnet_d,
                        FINAL_D=resnet_d,
                        norm=resnet_norm,
                        dropout=resnet_dropout,
                        OUT_DIM=OUT_DIM,
                    )
                    for _ in range(mixture_n)
                ]
            )

        self.input_vert_dropout = nn.Dropout(input_vert_dropout_p)
        self.input_edge_dropout = nn.Dropout(input_edge_dropout_p)

        self.out_std = out_std
        self.out_std_exp = False

        self.use_random_subsets = use_random_subsets
        self.mixture_num_obs_per = mixture_num_obs_per
        if embed_edges:
            self.edge_lin = nn.Linear(GS, GS)
        else:
            self.edge_lin = nn.Identity(GS)

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        if init_bias > 0:
                            nn.init.normal_(m.bias, 0, init_bias)
                        else:
                            nn.init.constant_(m.bias, 0)

    def forward(
        self,
        adj,
        vect_feat,
        input_mask,
        input_idx,
        adj_oh,
        return_g_features=False,
        also_return_g_features=False,
        **kwargs,
    ):
        G = self.edge_lin(adj.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        BATCH_N, MAX_N, F_N = vect_feat.shape

        if self.input_norm is not None:
            vect_feat = apply_masked_1d_norm(self.input_norm, vect_feat, input_mask)

        vect_feat = vect_feat * self.input_vert_dropout(input_mask).unsqueeze(-1)
        G = self.input_edge_dropout(G)

        G_features = self.gml(G, vect_feat, input_mask)
        if return_g_features:
            return G_features

        g_squeeze = G_features.squeeze(1)
        g_squeeze_flat = g_squeeze.reshape(-1, G_features.shape[-1])

        if self.resnet_out:
            x_1 = [
                m(g_squeeze_flat, input_mask.reshape(-1)).reshape(BATCH_N, MAX_N, -1)
                for m in self.mix_out
            ]
        else:
            x_1 = [m(g_squeeze) for m in self.mix_out]

        x_1 = torch.stack(x_1)

        x_1, std = bootstrap_perm_compute(
            x_1, input_idx, self.mixture_num_obs_per, training=self.training
        )

        ret = {"mu": x_1, "std": std}
        if also_return_g_features:
            ret["g_features"] = g_squeeze
        return ret


class GraphMatLayersDebug(nn.Module):
    def __init__(
        self,
        input_feature_n,
        output_features_n,
        resnet=False,
        GS=1,
        norm=None,
        force_use_bias=False,
        noise=1e-5,
        agg_func=None,
        layer_class="GraphMatLayerFast",
        intra_layer_dropout_p=0.0,
        layer_config={},
    ):
        super(GraphMatLayersDebug, self).__init__()

        self.gl = nn.ModuleList()
        self.dr = nn.ModuleList()
        self.resnet = resnet

        LayerClass = eval(layer_class)
        for li in range(len(output_features_n)):
            if li == 0:
                gl = LayerClass(
                    input_feature_n,
                    output_features_n[0],
                    noise=noise,
                    agg_func=agg_func,
                    GS=GS,
                    use_bias=not norm or force_use_bias,
                    **layer_config,
                )
            else:
                gl = LayerClass(
                    output_features_n[li - 1],
                    output_features_n[li],
                    noise=noise,
                    agg_func=agg_func,
                    GS=GS,
                    use_bias=not norm or force_use_bias,
                    **layer_config,
                )

            self.gl.append(gl)
            if intra_layer_dropout_p > 0:
                dr = nn.Dropout(intra_layer_dropout_p)
            else:
                dr = nn.Identity()
            self.dr.append(dr)

        self.norm = norm
        if self.norm is not None:
            if self.norm == "batch":
                Nlayer = MaskedBatchNorm1d
            elif self.norm == "layer":
                Nlayer = MaskedLayerNorm1d
            self.bn = nn.ModuleList([Nlayer(f) for f in output_features_n])

    def forward(self, G, x, input_mask=None):
        for gi, gl in enumerate(self.gl):
            x2 = gl(G, x)
            if self.norm:
                x2 = self.bn[gi](
                    x2.reshape(-1, x2.shape[-1]), input_mask.reshape(-1)
                ).reshape(x2.shape)

            x2 = x2 * self.dr[gi](input_mask).unsqueeze(-1)

            if self.resnet:
                if x.shape == x2.shape:
                    x3 = x2 + x
                else:
                    x3 = x2
            else:
                x3 = x2
            x = x3

        return x


class GraphMatLayerExpressionWNormAfter2(nn.Module):
    def __init__(
        self,
        C,
        P,
        GS=1,
        terms=[{"power": 1, "diag": False}],
        noise=1e-6,
        agg_func=None,
        use_bias=False,
        post_agg_nonlin=None,
        post_agg_norm=None,
        per_nonlin=None,
        dropout=0.0,
        cross_term_agg_func="sum",
        norm_by_neighbors=False,
    ):
        """ """

        super(GraphMatLayerExpressionWNormAfter2, self).__init__()

        self.pow_ops = nn.ModuleList()
        for t in terms:
            l = GraphMatLayerFastPow2(
                C,
                P,
                GS,
                mat_pow=t.get("power", 1),
                mat_diag=t.get("diag", False),
                noise=noise,
                use_bias=use_bias,
                nonlin=t.get("nonlin", per_nonlin),
                norm_by_neighbors=norm_by_neighbors,
                dropout=dropout,
            )
            self.pow_ops.append(l)

        self.post_agg_nonlin = post_agg_nonlin
        if self.post_agg_nonlin == "leakyrelu":
            self.r = nn.LeakyReLU()
        elif self.post_agg_nonlin == "relu":
            self.r = nn.ReLU()
        elif self.post_agg_nonlin == "sigmoid":
            self.r = nn.Sigmoid()
        elif self.post_agg_nonlin == "tanh":
            self.r = nn.Tanh()

        self.agg_func = agg_func
        self.cross_term_agg_func = cross_term_agg_func
        self.norm_by_neighbors = norm_by_neighbors
        self.post_agg_norm = post_agg_norm
        if post_agg_norm == "layer":
            self.pa_norm = nn.LayerNorm(P)

        elif post_agg_norm == "batch":
            self.pa_norm = nn.BatchNorm1d(P)

    def forward(self, G, x):
        BATCH_N, CHAN_N, MAX_N, _ = G.shape

        terms_stack = torch.stack([l(G, x) for l in self.pow_ops], dim=-1)

        if self.cross_term_agg_func == "sum":
            xout = torch.sum(terms_stack, dim=-1)
        elif self.cross_term_agg_func == "max":
            xout = torch.max(terms_stack, dim=-1)[0]
        elif self.cross_term_agg_func == "prod":
            xout = torch.prod(terms_stack, dim=-1)
        else:
            raise ValueError(f"unknown cross term agg func {self.cross_term_agg_func}")

        if self.agg_func is not None:
            xout = self.agg_func(xout, dim=0)

        if self.post_agg_norm is not None:
            xout = self.pa_norm(xout.reshape(-1, xout.shape[-1])).reshape(xout.shape)
        if self.post_agg_nonlin is not None:
            xout = self.r(xout)

        return xout
