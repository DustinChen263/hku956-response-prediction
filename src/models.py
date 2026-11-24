"""Small NumPy MLP with explicit validation and equal-task masked loss.

No internal random row split, no access to test targets, full-batch Adam.
"""
import copy
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def masked_loss_gradient(pred, truth):
    mask = np.isfinite(truth)
    counts = mask.sum(axis=0)
    active = counts > 0
    if not active.any():
        raise ValueError('No observed labels in loss')
    denom = np.maximum(counts, 1) * active.sum()
    error = np.where(mask, pred - np.nan_to_num(truth), 0.0)
    loss = float(np.sum(error ** 2 / denom))
    return loss, 2 * error / denom


class TargetScaler:
    def __init__(self, targets):
        self.targets = list(targets)

    def transform_raw(self, y):
        y = np.asarray(y, dtype=float).copy()
        for j, name in enumerate(self.targets):
            if name == 'rmssd_ms':
                if (y[:, j][np.isfinite(y[:, j])] < 0).any():
                    raise ValueError('Negative RMSSD target')
                y[:, j] = np.log1p(y[:, j])
        return y

    def fit(self, y):
        z = self.transform_raw(y)
        if (np.isfinite(z).sum(axis=0) == 0).any():
            raise ValueError('A target has no finite training labels')
        self.mean_ = np.nanmean(z, axis=0)
        self.scale_ = np.nanstd(z, axis=0)
        self.constant_targets_ = [self.targets[j] for j in np.flatnonzero(self.scale_ < 1e-10)]
        self.scale_[self.scale_ < 1e-10] = 1.0
        self.raw_scale_ = np.nanstd(y, axis=0)
        self.raw_scale_[self.raw_scale_ < 1e-10] = 1.0
        return self

    def transform(self, y):
        return (self.transform_raw(y) - self.mean_) / self.scale_

    def inverse(self, z):
        y = np.asarray(z) * self.scale_ + self.mean_
        for j, name in enumerate(self.targets):
            if name == 'rmssd_ms':
                y[:, j] = np.maximum(np.expm1(np.minimum(y[:, j], 50)), 0)
        return y


class SmallMLP:
    def __init__(self, hidden=32, l2=0.001, learning_rate=0.003, epochs=500, patience=40, min_delta=1e-5, seed=17):
        self.hidden, self.l2, self.lr = hidden, l2, learning_rate
        self.epochs, self.patience, self.min_delta, self.seed = epochs, patience, min_delta, seed

    def initialize(self, n_features, n_targets):
        rng = np.random.default_rng(self.seed)
        self.params = [rng.normal(0, np.sqrt(2 / n_features), (n_features, self.hidden)),
                       np.zeros(self.hidden), rng.normal(0, np.sqrt(1 / self.hidden), (self.hidden, n_targets)),
                       np.zeros(n_targets)]

    def predict(self, X):
        w1, b1, w2, b2 = self.params
        return np.maximum(X @ w1 + b1, 0) @ w2 + b2

    def loss_gradients(self, X, y):
        w1, b1, w2, b2 = self.params
        pre = X @ w1 + b1
        h = np.maximum(pre, 0)
        loss, delta = masked_loss_gradient(h @ w2 + b2, y)
        loss += 0.5 * self.l2 * (np.sum(w1 ** 2) + np.sum(w2 ** 2))
        dw2, db2 = h.T @ delta + self.l2 * w2, delta.sum(axis=0)
        dh = (delta @ w2.T) * (pre > 0)
        return loss, [X.T @ dh + self.l2 * w1, dh.sum(axis=0), dw2, db2]

    def fit(self, X, y, Xval, yval, participants=None):
        if not np.isfinite(X).all() or not np.isfinite(Xval).all():
            raise ValueError('Nonfinite model inputs')
        self.initialize(X.shape[1], y.shape[1])
        m = [np.zeros_like(p) for p in self.params]
        v = [np.zeros_like(p) for p in self.params]
        best, stale, saved = np.inf, 0, None
        self.history_ = []
        for step in range(1, self.epochs + 1):
            loss, gradients = self.loss_gradients(X, y)
            for i, grad in enumerate(gradients):
                m[i] = 0.9 * m[i] + 0.1 * grad
                v[i] = 0.999 * v[i] + 0.001 * grad ** 2
                self.params[i] -= self.lr * (m[i] / (1 - 0.9 ** step)) / (np.sqrt(v[i] / (1 - 0.999 ** step)) + 1e-8)
            pred = self.predict(Xval)
            score = validation_score(pred, yval, participants)
            self.history_.append({'epoch': step, 'train_loss': loss, 'validation_score': score})
            if not np.isfinite(loss) or not np.isfinite(score):
                raise ValueError('Training diverged')
            if score < best - self.min_delta:
                best, stale, saved = score, 0, copy.deepcopy(self.params)
                self.best_epoch_ = step
            else:
                stale += 1
            if stale >= self.patience:
                break
        if saved is None:
            raise ValueError('No valid checkpoint')
        self.params = saved
        self.parameter_count_ = sum(p.size for p in self.params)
        return self


def validation_score(pred, truth, groups=None):
    error = np.abs(pred - truth)
    # Standardized targets prevent averaging incomparable original units.
    if groups is None:
        return float(np.nanmean(np.nanmean(error, axis=0)))
    values = []
    for group in np.unique(groups):
        sub = error[groups == group]
        per_target = [np.mean(sub[np.isfinite(sub[:, j]), j]) for j in range(sub.shape[1]) if np.isfinite(sub[:, j]).any()]
        if per_target:
            values.append(np.mean(per_target))
    if not values:
        raise ValueError('No validation labels')
    return float(np.mean(values))


class ResponseModel:
    def __init__(self, kind, targets, config, seed=17):
        self.kind, self.targets, self.config, self.seed = kind, list(targets), config, seed

    def fit(self, X, y, Xval, yval, groupsval, songs=None):
        self.scaler = TargetScaler(self.targets).fit(y)
        z, zv = self.scaler.transform(y), self.scaler.transform(yval)
        self.xscaler = StandardScaler().fit(X)
        x, xv = self.xscaler.transform(X), self.xscaler.transform(Xval)
        self.pca = None
        components = self.config.get('pca_components')
        if components:
            components = min(int(components), x.shape[1], len(x) - 1)
            self.pca = PCA(n_components=components, svd_solver='full').fit(x)
            x, xv = self.pca.transform(x), self.pca.transform(xv)
        self.models, self.song_means = [], {}
        if self.kind in ('mean', 'song_mean'):
            self.parameter_count_ = len(self.targets)
            # Means in original units are the documented baseline; preserve log transform only for models.
            self.raw_mean_ = np.nanmean(y, axis=0)
            if self.kind == 'song_mean':
                if songs is None:
                    raise ValueError('song_mean requires training song ids')
                for song in np.unique(songs):
                    sub = y[songs == song]
                    means = np.array([np.mean(sub[np.isfinite(sub[:, j]), j]) if np.isfinite(sub[:, j]).any() else self.raw_mean_[j] for j in range(y.shape[1])])
                    self.song_means[str(song)] = means
        elif self.kind == 'ridge':
            for j in range(z.shape[1]):
                valid = np.isfinite(z[:, j])
                self.models.append(Ridge(alpha=self.config['alpha']).fit(x[valid], z[valid, j]))
            self.parameter_count_ = sum(m.coef_.size + 1 for m in self.models)
        elif self.kind in ('single_mlp', 'multi_mlp', 'av_mlp', 'matched_multi_mlp'):
            kwargs = {k: self.config[k] for k in ('hidden', 'l2', 'learning_rate', 'epochs', 'patience', 'min_delta')}
            if self.kind == 'single_mlp':
                for j in range(z.shape[1]):
                    model = SmallMLP(**kwargs, seed=self.seed).fit(x, z[:, [j]], xv, zv[:, [j]], groupsval)
                    self.models.append(model)
            else:
                self.models = [SmallMLP(**kwargs, seed=self.seed).fit(x, z, xv, zv, groupsval)]
            self.parameter_count_ = sum(m.parameter_count_ for m in self.models)
        else:
            raise ValueError(f'Unknown model: {self.kind}')
        return self

    def predict(self, X, songs=None):
        if self.kind == 'mean':
            return np.tile(self.raw_mean_, (len(X), 1))
        if self.kind == 'song_mean':
            if songs is None:
                raise ValueError('song_mean is diagnostic and requires candidate song IDs')
            return np.stack([self.song_means.get(str(s), self.raw_mean_) for s in songs])
        x = self.xscaler.transform(X)
        if self.pca is not None:
            x = self.pca.transform(x)
        if self.kind in ('ridge', 'single_mlp'):
            pred = np.column_stack([m.predict(x).reshape(-1) for m in self.models])
        else:
            pred = self.models[0].predict(x)
        return self.scaler.inverse(pred)
