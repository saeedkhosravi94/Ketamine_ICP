import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier


class SASMOTE:
    def __init__(
        self,
        minority_class=None,
        k_neighbors=5,
        random_state=None,
        rf_n_estimators=100,
        rf_max_depth=None,
    ):
        self.minority_class = minority_class
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rng = np.random.default_rng(random_state)

    def _validate_input(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")

        classes, counts = np.unique(y, return_counts=True)

        if len(classes) != 2:
            raise ValueError("SASMOTE currently supports binary classification only.")

        return X, y, classes, counts

    def _get_class_info(self, y, classes, counts):
        minority_class = (
            self.minority_class
            if self.minority_class is not None
            else classes[np.argmin(counts)]
        )
        majority_class = classes[classes != minority_class][0]
        return minority_class, majority_class

    def _split_classes(self, X, y, minority_class, majority_class):
        X_min = X[y == minority_class]
        X_maj = X[y == majority_class]
        return X_min, X_maj

    def _fit_neighbors(self, X_min):
        n_min = len(X_min)
        k = min(self.k_neighbors + 1, n_min)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X_min)
        return nn.kneighbors(X_min, return_distance=False)

    def _visible_neighbors(self, x_idx, X_min, neighbor_indices):
        idxs = neighbor_indices[x_idx][1:]
        x = X_min[x_idx]
        visible = []

        for y_idx in idxs:
            y_pt = X_min[y_idx]
            is_visible = True

            for z_idx in idxs:
                if z_idx == y_idx:
                    continue
                z_pt = X_min[z_idx]
                v1 = x - z_pt
                v2 = y_pt - z_pt
                if np.dot(v1, v2) < 0:
                    is_visible = False
                    break

            if is_visible:
                visible.append(y_idx)

        return visible

    def _build_visible_map(self, X_min, neighbor_indices):
        visible_map = [
            self._visible_neighbors(i, X_min, neighbor_indices)
            for i in range(len(X_min))
        ]
        valid_seed_indices = [i for i, v in enumerate(visible_map) if len(v) > 0]
        return visible_map, valid_seed_indices

    def _generate_synthetic_samples(self, X_min, visible_map, valid_seed_indices, n_to_generate):
        synthetic_samples = []

        while len(synthetic_samples) < n_to_generate:
            x_idx = self.rng.choice(valid_seed_indices)
            y_idx = self.rng.choice(visible_map[x_idx])

            x_pt = X_min[x_idx]
            y_pt = X_min[y_idx]

            lam = self.rng.random()
            s = x_pt + lam * (y_pt - x_pt)
            synthetic_samples.append(s)

        return np.asarray(synthetic_samples, dtype=np.float32)

    def _train_single_inspector(self, X_min, X_maj, minority_class, majority_class, y_dtype, inspector_idx):
        n_min = len(X_min)
        n_maj = len(X_maj)
        maj_indices = np.arange(n_maj)

        replace = n_maj < n_min
        sampled_maj_idx = self.rng.choice(maj_indices, size=n_min, replace=replace)
        X_maj_subset = X_maj[sampled_maj_idx]

        X_train_inspector = np.vstack([X_min, X_maj_subset])
        y_train_inspector = np.concatenate([
            np.full(n_min, minority_class, dtype=y_dtype),
            np.full(n_min, majority_class, dtype=y_dtype)
        ])

        rf = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            random_state=None if self.random_state is None else self.random_state + inspector_idx,
            n_jobs=-1
        )
        rf.fit(X_train_inspector, y_train_inspector)
        return rf

    def _filter_synthetic_samples(self, X_syn, X_min, X_maj, minority_class, majority_class, y_dtype):
        n_min = len(X_min)
        n_maj = len(X_maj)
        n_inspectors = max(1, n_maj // n_min)

        inspector_votes_majority = np.zeros(len(X_syn), dtype=int)

        for i in range(n_inspectors):
            rf = self._train_single_inspector(
                X_min, X_maj, minority_class, majority_class, y_dtype, i
            )
            preds = rf.predict(X_syn)
            inspector_votes_majority += (preds == majority_class).astype(int)

        uncertainty_scores = inspector_votes_majority / n_inspectors
        keep_mask = uncertainty_scores < 0.5

        X_syn_filtered = X_syn[keep_mask]
        y_syn_filtered = np.full(len(X_syn_filtered), minority_class, dtype=y_dtype)

        return X_syn_filtered, y_syn_filtered

    def _combine_data(self, X, y, X_syn_filtered, y_syn_filtered):
        X_res = np.vstack([X, X_syn_filtered]).astype(np.float32)
        y_res = np.concatenate([y, y_syn_filtered])
        return X_res, y_res

    def fit_resample(self, X, y):
        X, y, classes, counts = self._validate_input(X, y)
        minority_class, majority_class = self._get_class_info(y, classes, counts)
        X_min, X_maj = self._split_classes(X, y, minority_class, majority_class)

        n_min = len(X_min)
        n_maj = len(X_maj)

        if n_min < 2:
            raise ValueError("Not enough minority samples to apply SASMOTE.")

        n_to_generate = n_maj - n_min
        if n_to_generate <= 0:
            return X.copy(), y.copy()

        neighbor_indices = self._fit_neighbors(X_min)
        visible_map, valid_seed_indices = self._build_visible_map(X_min, neighbor_indices)

        if not valid_seed_indices:
            return X.copy(), y.copy()

        X_syn = self._generate_synthetic_samples(
            X_min, visible_map, valid_seed_indices, n_to_generate
        )

        X_syn_filtered, y_syn_filtered = self._filter_synthetic_samples(
            X_syn, X_min, X_maj, minority_class, majority_class, y.dtype
        )

        return self._combine_data(X, y, X_syn_filtered, y_syn_filtered)