import numpy as np
from src.models import SmallMLP, TargetScaler, masked_loss_gradient, ResponseModel


def test_equal_task_masked_loss():
    pred = np.zeros((3,2))
    truth = np.array([[1,2],[1,np.nan],[1,np.nan]])
    loss, gradient = masked_loss_gradient(pred,truth)
    assert np.isclose(loss, 2.5)  # (mean([1,1,1]) + mean([4])) / 2
    assert gradient[1,1] == 0
    assert np.isclose(gradient[:,0].sum(),-1)
    assert gradient[0,1] == -2


def test_mlp_gradients_finite_difference():
    rng=np.random.default_rng(17)
    x=rng.normal(size=(5,3)); y=rng.normal(size=(5,2)); y[1,0]=np.nan
    m=SmallMLP(hidden=4,l2=.01)
    m.initialize(3,2)
    _, grads=m.loss_gradients(x,y)
    eps=1e-6
    for p, grad in zip(m.params,grads):
        for index in np.ndindex(p.shape):
            old=p[index]
            p[index]=old+eps; high=m.loss_gradients(x,y)[0]
            p[index]=old-eps; low=m.loss_gradients(x,y)[0]
            p[index]=old
            assert np.isclose((high-low)/(2*eps),grad[index],atol=1e-5)


def test_target_scaler_round_trip_and_constant():
    y=np.array([[20.,1.],[40.,1.],[np.nan,1.]])
    s=TargetScaler(['rmssd_ms','arousal']).fit(y)
    assert np.allclose(s.inverse(s.transform(y)),y,equal_nan=True)
    assert s.constant_targets_ == ['arousal']


def test_scaler_fits_only_training_values():
    x=np.array([[1.,2.],[2.,3.],[3.,4.]])
    xv=np.array([[1000.,2000.],[2000.,3000.]])
    y=np.array([[1.],[2.],[3.]])
    model=ResponseModel('ridge',['arousal'],{'alpha':1.,'pca_components':None})
    model.fit(x,y,xv,np.array([[0.],[1.]]),np.array(['v','v']))
    assert np.allclose(model.xscaler.mean_,[2,3])
    assert np.allclose(model.scaler.mean_,[2])


def test_small_network_learns_with_explicit_validation():
    rng=np.random.default_rng(9)
    x=rng.normal(size=(40,2)); y=(x[:,0]*.8-x[:,1]*.3)[:,None]
    m=SmallMLP(hidden=12,epochs=300,patience=50,learning_rate=.01,l2=.0001)
    m.fit(x[:30],y[:30],x[30:],y[30:],np.array(['v']*10))
    assert np.mean(abs(m.predict(x[30:])-y[30:])) < .25


def test_capacity_matching_uses_all_single_task_parameters():
    from src.train import candidates
    cfg={'learning_rate':.003,'epochs':10,'patience':3,'min_delta':1e-5,'pca_components':None,
         'hidden_sizes':[16],'l2_values':[.001]}
    params=candidates('matched_multi_mlp',cfg,37,3)[0]
    single_total=3 * (37*16+16+16+1)
    shared_total=params['hidden']*(37+3+1)+3
    assert abs(single_total-shared_total) <= (37+3+1)/2
