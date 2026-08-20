# import bilby
# from bilby.core.sampler.base_sampler import Sampler

# class BilbyFlow(Sampler):
#     default_kwargs = dict(npe_dir=None, n_samples=10000, prior_swap=True,
#                           npool=1, checkpoint_path=None, standardiser_path=None)
#     sampler_name = "bilbyflow"

#     def run_sampler(self):
#         # self.likelihood carries .interferometers -> build x
#         # load model from kwargs["npe_dir"], sample, reweight
#         # then populate:
#         self.result.samples = samples_phys          # (n, ndim) in search-param order
#         self.result.log_likelihood_evaluations = log_likelihood
#         self.result.log_evidence = log_Z            # from logsumexp(log_w) - log N
#         self.result.meta_data["kish_ess"] = 
#         return self.result