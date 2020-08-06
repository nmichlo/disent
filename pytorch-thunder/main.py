import torch


# ========================================================================= #
# main                                                                   #
# ========================================================================= #

# NEEDS: Topological sorting - valid topological order is a valid evaluation order of the dependency graph
# https://en.wikipedia.org/wiki/Topological_sorting
# https://en.wikipedia.org/wiki/Dependency_graph

# pythonflow by Spotify looks like a good library for this.
# https://pythonflow.readthedocs.io/en/latest/guide.html
# https://github.com/spotify/pythonflow


# # VAE DEFINITION
#
# encoder = Encoder()
# reparameterize = Reparameterize()
# decoder = Decoder()
#
# vae = ThunderModule()
# vae = vae[encoder]
# vae = vae[reparameterize]
# vae = vae[decoder]
#
# vae = decoder[reparametrize[encoder]]

# VAE DEFINITION
# x = Input('x')
# z_params = encoder.accept(x)
# reparamed = reparameterize.accept(z_params)
# decoded = decoder.accept(reparamed)
#
# loss = GaussianKlLoss(z_params) + ReconstructionLoss(x, decoded)

# ADAVAE DEFINITION - COULD THIS WORK?
#
# encoder_stage = (encoder + encoder)
# adavae = adaptive_average(adavae)
# adavae = (sample  +  sample)(adavae)
# adavae = (decoder + decoder)(adavae)

# loss = decoder + reparameterize

# I BASICALLY CAME UP WITH KERAS...
# x0, x1 = Input(x0), Input(x1)
# # begin
# z0_params = Node(encoder).accept(x0)
# z1_params = Node(encoder).accept(x1)
# # average
# z0_params_mean, z1_params_mean = Node(AdaptiveAverage).accept(z0_params, z1_params)
# # reparametrise
# z0 = Node(reparameterize).accept(z0_params_mean)
# z1 = Node(reparameterize).accept(z1_params_mean)
# # do stuff
# x0_recon = Node(decoder).accept(z0)
# x1_recon = Node(decoder).accept(z1)
#
# adavae = Model(inputs=[x0, x1], outputs=[z0_params_mean, z1_params_mean, x0_recon, x1_recon])


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
