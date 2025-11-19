# -*- coding: utf-8 -*-
"""
Created on 2024-05-15 (Wed) 16:09:44

@author: I.Azuma
"""
# Copyright 2019-2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This code is modified from this git repo: https://github.com/xunzheng/notears

@inproceedings{zheng2020learning,
    author = {Zheng, Xun and Dan, Chen and Aragam, Bryon and Ravikumar, Pradeep and Xing, Eric P.},
    booktitle = {International Conference on Artificial Intelligence and Statistics},
    title = {{Learning sparse nonparametric DAGs}},
    year = {2020}
}
"""
import logging
from typing import Iterable, List, Tuple, Union

import numpy as np
import scipy.optimize as sopt
import torch
from sklearn.base import BaseEstimator
from torch import nn

from network_models.dual_estimator.dist_type._base import DistTypeBase
from network_models.dual_estimator.nonlinear import LocallyConnected

class DiseaseMLP(nn.Module, BaseEstimator):
    def __init__(
        self,
        n_features: int,
        dist_types: List[DistTypeBase],
        use_bias: bool = False,
        use_gpu: bool = True,
        hidden_layer_units: Iterable[int] = (0,),
        bounds: List[Tuple[int, int]] = None,
        ls_gamma: float = 0.5,
        lasso_beta: float = 0.0,
        ridge_beta: float = 0.0,
        nonlinear_clamp: float = 1e-2,
        device: str = "cuda",
    ):
        super().__init__()

        self.device = device

        # cast to list for later concat.
        self.dims = (
            [n_features] + list(hidden_layer_units) + [1]
            if hidden_layer_units[0]
            else [n_features, 1]
        )
        # dag_layer: initial linear layer
        self.dag_layer = nn.Linear(
            self.dims[0], self.dims[0] * self.dims[1], bias=use_bias
        ).float()

        self.dag_layer.to(self.device)  # Send DAG layer to device
        nn.init.zeros_(self.dag_layer.weight)

        if use_bias:
            nn.init.zeros_(self.dag_layer.bias)

        # loc_lin_layer: local linear layers
        layers = [
            LocallyConnected(
                self.dims[0], input_features, output_features, bias=use_bias
            ).float()
            for input_features, output_features in zip(self.dims[1:-1], self.dims[2:])
        ]
        self._loc_lin_layer_weights = nn.ModuleList(layers)

        for layer in layers:
            layer.reset_parameters()
            layer.to(self.device)

        # set the bounds as an attribute on the weights object
        self.dag_layer.weight.bounds = bounds
        # set the dist types
        self.dist_types = dist_types
        # type the adjacency matrix
        self.adj_d = None
        self.adj_d_mean_effect = None
    
    @property
    def _logger(self):
        return logging.getLogger(self.__class__.__name__)

    @property
    def dag_layer_bias(self) -> Union[torch.Tensor, None]:
        """
        dag_layer bias is the bias of the first fully connected layer which determines the causal structure.

        Returns:
            dag_layer bias if use_bias is True, otherwise None
        """
        return self.dag_layer.bias

    @property
    def dag_layer_weight(self) -> torch.Tensor:
        """
        dag_layer weight is the weight of the first fully connected layer which determines the causal structure.

        Returns:
            dag_layer weight
        """
        return self.dag_layer.weight

    @property
    def loc_lin_layer_weights(self) -> torch.Tensor:
        """
        loc_lin_layer weights are the weight of hidden layers after the first fully connected layer.

        Returns:
            loc_lin_layer weights
        """
        return self._loc_lin_layer_weights
    
    @property
    def bias(self) -> Union[np.ndarray, None]:
        """
        Get the vector of feature biases

        Returns:
            bias vector if use_bias is True, otherwise None
        """
        bias = self.dag_layer_bias
        return bias if bias is None else bias.cpu().detach().numpy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n, d] -> [n, d]
        """
        Feed forward calculation for the model.

        Args:
            x: input torch tensor

        Returns:
            output tensor from the model
        """
        x = self.dag_layer(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]

        for output_dim, layer in zip(self.dims[1:], self.loc_lin_layer_weights):
            x = torch.sigmoid(x)  # [n, d, m1]
            x = nn.LayerNorm(
                output_dim,
                eps=self.nonlinear_clamp,
                elementwise_affine=True,
            )(x)
            x = layer(x)  # [n, d, m2]

        x = x.squeeze(dim=2)  # [n, d]
        return x
    
    def _calculate_adj(self, X: torch.Tensor, mean_effect: bool) -> torch.Tensor:
        """
        Calculate the adjacency matrix.

        For the linear case, this is just dag_layer_weight.
        For the nonlinear case, approximate the relationship using the gradient of X_hat wrt X.
        """
        # for the linear case, save compute by just returning the dag_layer weights
        if len(self.dims) <= 2:
            adj = (
                self.dag_layer_weight.T
                if mean_effect
                else torch.abs(self.dag_layer_weight.T)
            )
            return adj

        _, n_features = X.shape

        # get the data X and reconstruction X_hat
        X = X.clone().requires_grad_()
        X_hat = self(X).sum(dim=0)  # shape = (n_features,)

        # iterate over sums of reconstructed features
        adj = []

        for j in range(n_features):

            # calculate the gradient of X_hat wrt X
            ddx = torch.autograd.grad(X_hat[j], X, create_graph=True)[0]

            if mean_effect:
                # get the average effect
                adj.append(ddx.mean(axis=0).unsqueeze(0))
            else:
                # otherwise, use the average L1 of the gradient as the W
                adj.append(torch.abs(ddx).mean(dim=0).unsqueeze(0))

        adj = torch.cat(adj, dim=0)

        # transpose to get the adjacency matrix
        return adj.T


# Problem in pytorch 1.6 (_forward_unimplemented), fixed in next release:
class Dual_NotearsMLP(BaseEstimator):
    """
    Class for NOTEARS MLP (Multi-layer Perceptron) model.
    The model weights consist of dag_layer and loc_lin_layer weights respectively.
    dag_layer weight is the weight of the first fully connected layer which determines the causal structure.
    loc_lin_layer weights are the weight of hidden layers after the first fully connected layer
    """
    def __init__(
        self,
        n_features: int,
        dist_types_h: List[DistTypeBase],
        dist_types_d: List[DistTypeBase],
        use_bias: bool = False,
        use_gpu: bool = True,
        hidden_layer_units: Iterable[int] = (0,),
        bounds_h: List[Tuple[int, int]] = None,
        bounds_d: List[Tuple[int, int]] = None,
        ls_gamma: float = 0.5,
        lasso_beta: float = 0.0,
        ridge_beta: float = 0.0,
        nonlinear_clamp: float = 1e-2,
    ):
        avail_device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        self.device = torch.device(avail_device)
        self.dist_types_h = dist_types_h
        self.dist_types_d = dist_types_d
        self.ls_gamma = ls_gamma
        self.lasso_beta = lasso_beta
        self.ridge_beta = ridge_beta

        self.disease_branch = DiseaseMLP(
            n_features=n_features,
            dist_types=dist_types_d,
            hidden_layer_units=hidden_layer_units,
            ls_gamma=ls_gamma,
            lasso_beta=lasso_beta,
            ridge_beta=ridge_beta,
            bounds=bounds_d,
            use_bias=use_bias,
            use_gpu=use_gpu,
            nonlinear_clamp=nonlinear_clamp,
            device=self.device,
        )

        self.healthy_branch = DiseaseMLP(  # FIXME: Prepare independent branch
            n_features=n_features,
            dist_types=dist_types_h,
            hidden_layer_units=hidden_layer_units,
            ls_gamma=ls_gamma,
            lasso_beta=lasso_beta,
            ridge_beta=ridge_beta,
            bounds=bounds_h,
            use_bias=use_bias,
            use_gpu=use_gpu,
            nonlinear_clamp=nonlinear_clamp,
            device=self.device,
        )


    def fit(
        self,
        xd: np.ndarray,  # disease
        xh: np.ndarray,  # healthy
        #w: np.ndarray,
        w_binary_mask: bool = True,
        max_iter: int = 100,
        h_tol: float = 1e-8,
        rho_max: float = 1e16,
    ):
        """
        Fit NOTEARS MLP model using the input data x

        Args:
            x: 2d numpy array input data, axis=0 is data rows, axis=1 is data columns. Data must be row oriented.
            max_iter: max number of dual ascent steps during optimisation.
            h_tol: exit if h(w) < h_tol (as opposed to strict definition of 0).
            rho_max: to be updated
        """
        rho, alpha, h = 1.0, 0.0, np.inf
        Xd_torch = torch.from_numpy(xd).float().to(self.device)
        Xh_torch = torch.from_numpy(xh).float().to(self.device)
        #W_torch = torch.from_numpy(w).float().to(self.device)
        """ TODO: future work
        if w_binary_mask:
            binary_mask = torch.where(W_torch==0,0.0,1.0).to(self.device)  # NOTE: Specify elements to consider for loss
        else:
            binary_mask = torch.ones(W_torch.shape).to(self.device)
        """
        for n_iter in range(max_iter):
            rho, alpha, h = self._dual_ascent_step(Xd_torch, Xh_torch, rho, alpha, h, rho_max)

            if h <= h_tol or rho >= rho_max:
                break
            if n_iter == max_iter - 1 and h > h_tol:
                self._logger.warning(
                    "Failed to converge. Consider increasing max_iter."
                )

        # calculate the adjacency matrix after the fitting is finished
        self.adj_d = (
            self.disease_branch._calculate_adj(Xd_torch, mean_effect=False).cpu().detach().numpy()
        )
        self.adj_d_mean_effect = (
            self.disease_branch._calculate_adj(Xd_torch, mean_effect=True).cpu().detach().numpy()
        )
        self.adj_h = (
            self.healthy_branch._calculate_adj(Xh_torch, mean_effect=False).cpu().detach().numpy()
        )
        self.adj_h_mean_effect = (
            self.healthy_branch._calculate_adj(Xh_torch, mean_effect=True).cpu().detach().numpy()
        )


    def _dual_ascent_step(
        self, Xd: torch.Tensor, Xh: torch.Tensor, rho: float, alpha: float, h: float, rho_max: float
    ) -> Tuple[float, float, float]:
        """
        Perform one step of dual ascent in augmented Lagrangian.

        Args:
            X: input tensor data.
            rho: max number of dual ascent steps during optimisation.
            alpha: exit if h(w) < h_tol (as opposed to strict definition of 0).
            h: DAGness of the adjacency matrix
            rho_max: to be updated

        Returns:
            rho, alpha and h
        """

        def _get_flat_grad(params: List[torch.Tensor]) -> np.ndarray:
            """
            Get flatten gradient vector from the parameters of the model

            Args:
                params: parameters of the model

            Returns:
                flatten gradient vector in numpy form
            """
            views = [
                p.data.new(p.data.numel()).zero_()
                if p.grad is None
                else p.grad.data.to_dense().view(-1)
                if p.grad.data.is_sparse
                else p.grad.data.view(-1)
                for p in params
            ]
            return torch.cat(views, 0).cpu().detach().numpy()

        def _get_flat_bounds(
            params: List[torch.Tensor],
        ) -> List[Tuple[Union[None, float]]]:
            """
            Get bound constraint for each parameter in flatten vector form from the parameters of the model

            Args:
                params: parameters of the model

            Returns:
                flatten vector of bound constraints for each parameter in numpy form
            """
            bounds = []

            for p in params:
                try:
                    b = p.bounds
                except AttributeError:
                    b = [(None, None)] * p.numel()
                bounds += b

            return bounds

        def _get_flat_params(params: List[torch.Tensor]) -> np.ndarray:
            """
            Get parameters in flatten vector from the parameters of the model

            Args:
                params: parameters of the model

            Returns:
                flatten parameters vector in numpy form
            """
            views = [
                p.data.to_dense().view(-1) if p.data.is_sparse else p.data.view(-1)
                for p in params
            ]
            return torch.cat(views, 0).cpu().detach().numpy()

        def _update_params_from_flat(
            params: List[torch.Tensor], flat_params: np.ndarray
        ):
            """
            Update parameters of the model from the parameters in the form of flatten vector

            Args:
                params: parameters of the model
                flat_params: parameters in the form of flatten vector
            """
            offset = 0
            flat_params_torch = (
                torch.from_numpy(flat_params)
                .to(torch.get_default_dtype())
                .to(self.device)
            )
            for p in params:
                n_params = p.numel()
                # view_as to avoid deprecated pointwise semantics
                p.data = flat_params_torch[offset : offset + n_params].view_as(p.data)
                offset += n_params

        def _func(flat_params: np.ndarray) -> Tuple[float, np.ndarray]:
            """
            Objective function that the NOTEARS algorithm tries to minimise.

            Args:
                flat_params: parameters to be optimised to minimise the objective function

            Returns:
                Loss and gradient
            """
            _update_params_from_flat(params, flat_params)
            optimizer.zero_grad()

            n_features = Xd.shape[1]
            Xd_hat = self.disease_branch.forward(Xd)
            Xh_hat = self.healthy_branch.forward(Xh)  # NOTE: share the parameter ??
            adj_d = self.disease_branch._calculate_adj(Xd,mean_effect=False)  # specific adjacency matrix
            adj_h = self.healthy_branch._calculate_adj(Xh,mean_effect=False)  # common adjacency matrix
            h_val_d = self._h_func(self.disease_branch)
            h_val_h = self._h_func(self.healthy_branch)
            h_val = h_val_d + h_val_h
            loss = 0.0

            # sum the losses across all dist types
            """
            for dist_type in self.dist_types:
                loss = loss + (self.ls_gamma*dist_type.loss_d(Xd, Xd_hat, adj_h)) + ((1-self.ls_gamma)*dist_type.loss_h(Xh, Xh_hat))
            """
            for dist_type_h in self.dist_types_h:
                loss = loss + (1-self.ls_gamma)*dist_type_h.loss_h(Xh, Xh_hat)
            for dist_type_d in self.dist_types_d:
                loss = loss + self.ls_gamma*dist_type_d.loss_d(Xd, Xd_hat, adj_h)

            lagrange_penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            # NOTE: both the l2 and l1 regularization are NOT applied to the bias parameters
            l2_reg = 0.5 * self.ridge_beta * self._l2_reg(n_features)
            l1_reg = self.lasso_beta * self._l1_reg(n_features)

            primal_obj = loss + lagrange_penalty + l2_reg + l1_reg
            primal_obj.backward()
            loss = primal_obj.item()

            flat_grad = _get_flat_grad(params)
            return loss, flat_grad.astype("float64")

        #optimizer = torch.optim.Optimizer(self.disease_branch.parameters(), {})
        optimizer = torch.optim.Optimizer(list(self.disease_branch.parameters()) + list(self.healthy_branch.parameters()), {})
        params = optimizer.param_groups[0]["params"]
        flat_params = _get_flat_params(params)
        bounds = _get_flat_bounds(params)
        h_new = np.inf

        while (rho < rho_max) and (h_new > 0.25 * h or h_new == np.inf):
            # Magic
            sol = sopt.minimize(
                _func,
                flat_params,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
            )
            _update_params_from_flat(params, sol.x)
            h_new_d = self._h_func(self.disease_branch).item()
            h_new_h = self._h_func(self.healthy_branch).item()
            h_new = h_new_d + h_new_h

            if h_new > 0.25 * h:
                rho *= 10

        alpha += rho * h_new
        return rho, alpha, h_new

    def _h_func(self, branch) -> torch.Tensor:
        """
        Constraint function of the NOTEARS algorithm.
        Constrain 2-norm-squared of dag_layer weights of the model along m1 dim to be a DAG

        Returns:
            DAGness of the adjacency matrix
        """
        #d = self.disease_branch.dims[0]
        d = branch.dims[0]

        # only consider the dag_layer for h(W) for compute efficiency
        #dag_layer_weight = self.disease_branch.dag_layer_weight.view(d, -1, d)  # [j, m1, i]
        dag_layer_weight = branch.dag_layer_weight.view(d, -1, d)  # [j, m1, i]
        square_weight_mat = torch.sum(
            dag_layer_weight * dag_layer_weight, dim=1
        ).t()  # [i, j]

        # modify the h(W) matrix to deal with expanded columns
        original_idxs = []

        for dist_type in branch.dist_types:
            # modify the weight matrix to prevent spurious cycles with expended columns
            square_weight_mat = dist_type.modify_h(square_weight_mat)
            # gather the original idxs
            original_idxs.append(dist_type.idx)

        # original size is largest original index
        original_size = np.max(original_idxs) + 1
        # subselect the top LH corner of matrix which corresponds to original data
        square_weight_mat = square_weight_mat[:original_size, :original_size]
        # update d and d_torch to match the new matrix size
        d = square_weight_mat.shape[0]
        d_torch = torch.tensor(d).to(self.device)

        # h = trace_expm(a) - d  # (Zheng et al. 2018)
        characteristic_poly_mat = (
            torch.eye(d).to(self.device) + square_weight_mat / d_torch
        )  # (Yu et al. 2019)
        polynomial_mat = torch.matrix_power(characteristic_poly_mat, d - 1)
        h = (polynomial_mat.t() * characteristic_poly_mat).sum() - d
        return h

    def _l1_reg(self, n_features: int) -> torch.Tensor:
        """
        Take average l1 of all weight parameters of the model.
        NOTE: regularisation needs to be scaled up by the number of features
        because the loss scales with feature number.

        Returns:
            l1 regularisation term.
        """
        return torch.mean(torch.abs(self.disease_branch.dag_layer_weight)) * n_features

    def _l2_reg(self, n_features: int) -> torch.Tensor:
        """
        Take average 2-norm-squared of all weight parameters of the model.
        NOTE: regularisation needs to be scaled up by the number of features
        because the loss scales with feature number.

        Returns:
            l2 regularisation term.
        """
        reg = 0.0
        reg += torch.sum(self.disease_branch.dag_layer_weight**2)

        for layer in self.disease_branch.loc_lin_layer_weights:
            reg += torch.sum(layer.weight**2)

        # calculate the total number of elements used in the above sums
        n_elements = self.disease_branch.dag_layer_weight.numel()

        for layer in self.disease_branch.loc_lin_layer_weights:
            n_elements = n_elements + layer.weight.numel()

        return reg / n_elements * n_features

    
