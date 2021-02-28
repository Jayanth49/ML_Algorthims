# GAN

* It is just 2 Player Game,Discriminator vs Generator
* We want to train such that Both performs well and Generator can fool Discriminator.
* Parameters of Discriminator: D
* Parameters of Generator: G
* The discriminator wishes to minimize JD(D,G) and must do so while controlling only D. The generator wishes to minimize
  JG(D,G) and must do so while controlling only G.
* The solution to a game is a Nash equilibrium. Here, we use the terminology of local differential Nash equilibria.That is optimal G,D that is a local minimum of JD wrt to D and JG wrt to G.
* **THE GENERATOR:**
  * The generator is simply a differentiable function G. When z is sampled from some simple prior distribution, G(z) yields a sample of x drawn from p_model.
  * 