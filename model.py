import jax
from jax import numpy as jnp
import chex
import optax
from functools import partial
from typing import Any, Callable

from flax import linen as nn
import flax
from flax.training.train_state import TrainState

from buffer import Transition


class DQNTrainingArgs:
    gamma: float = 0.99  # discounting factor in MDP
    learning_rate: float = 2.5e-4  # learning rate for DQN parameter optimization
    target_update_every: int = 512  # the target network update frequency (per training steps)
    fifo_buffer_size: int = 10000  # the total size of the replay buffer
    buffer_prefill: int = 10000  # the number of transitions to prefill the replay buffer with.
    train_batch_size: int = 128  # the batch size used in training
    start_eps: float = 1.0  # epsilon (of epsilon-greedy action selection) in the beginning of the training
    end_eps: float = 0.05  # epsilon (of epsilon-greedy action selection) in the end of the training
    epsilon_decay_steps: int = 25_000  # how many steps to decay epsilon over
    sample_budget: int = 250_000  # the total number of environment transitions to train our agent over
    eval_env_steps: int = 5000  # total number of env steps to evaluate the agent over
    eval_environments: int = 10  # how many parallel environments to use in evaluation
    # say we do 1 training step per N "environment steps" (i.e. per N sampled MDP transitions); 
    # also, say train batch size in this step is M (in the number of MDP transitions).
    # train_intensity is the desired fraction M/N.
    # i.e. the ratio of "replayed" transitions to sampled transitions
    # the higher this number is, the more intense experience replay will be.
    # to keep the implementation simple, we don't allow to make this number
    # bigger that the batch size but it can be an arbitrarily small positive number
    train_intensity: float = 8.0


class DQN(nn.Module):
    n_actions: int
    state_shape: list[int]

    @nn.compact
    def __call__(self, state: '[batch, *state_shape]') -> '[batch, n_actions]':
        """ This function defines the forward pass of Deep Q-Network.

        Note that the expected format of convolutional layers is [B, H, W, C]
        Where B - batch dimension, H, W - height and width dimensions respectively
        C - channels dimension

        Args:
            state: dtype float32, shape [batch, *state_shape] a batch of states of MDP
        Returns:
            array containing Q-values for each action, its shape is [batch, n_actions]
        """
        batch = state.shape[0]
        ################
        new_dim = int(state.shape[1] ** 0.5)  # Try to reshape into a square-like shape
        x = jnp.reshape(state, (batch, new_dim, new_dim, 1))  # Reshape to (batch_size, new_dim, new_dim, 1)
        print(f"State shape: {state.shape}, state shape is {self.state_shape} ")
        x = state
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)

        ################

        return x


DQNParameters = flax.core.frozen_dict.FrozenDict


class DQNTrainState(TrainState):
    # Note that `apply_fn`, `params`, and `tx` are inherited from TrainState
    target_params: DQNParameters


@chex.dataclass(frozen=True)
class DQNAgent:
    dqn: DQN  # the Deep Q-Network instance of the agent
    initialize_agent_state: Callable[[Any], DQNTrainState]
    """initialize_agent_state:
    creates the training state for our DQN agent.
    """
    select_action: Callable[[DQN, chex.PRNGKey, DQNParameters, chex.Array, chex.Array], chex.Array]
    """select_action:
    This function takes a random key of jax, a Deep Q-Network instance and its parameters
    as well as the state of MDP and the epsilon parameter and performs the action selection
    with an epsilon greedy strategy. Note that this function should be vmap-able.
    """
    compute_loss: Callable[[DQN, DQNParameters, DQNParameters, Transition, float], chex.Array]
    """compute_loss:
    This function computes the Deep Q-Network loss. It takes as an input the DQN object,
    the current parameters of the DQN agent and target parameters of the 
    DQN agent. Additionally it accepts the `Transition` object (see buffer.py for definition) and
    the gamma discounting factor. 
    """
    update_target: Callable[[DQNTrainState], DQNTrainState]
    """update_target: 
    performs the target network parameters update making the latter equal to the current parameters.
    """


def select_action(dqn: DQN, rng: chex.PRNGKey, params: DQNParameters, state: chex.Array,
                  epsilon: chex.Array) -> chex.Array:
    """ selects an action according to the epsilon greedy strategy

    Args:
        rng (chex.PRNGKey): random number generator
        dqn (DQN): the Deep Q-Network model object
        params: (DQNParameters): the parameters of DQN
        state (chex.Array, dtype float32, shape [*state_shape]): the state to infer q-values from
        epsilon (chex.Array, dtype float32, shape ()): the epsilon parameter in the epison-greedy
            action selection strategy. Note that this argument is a scalar array of shape ()
    Returns:
        action (chex.Array, dtype int32, shape ()): selected action.
        You should assume that rng is a "single" key, i.e. as if it was
        returned by `jax.random.key(42)` and implement this function accordingly.

        Note: remember that the output of NN is [batch, dims]
    """
    ################
    ## YOUR CODE GOES HERE
    q_values = dqn.apply(params, state)
    key, subkey = jax.random.split(rng, 2)
    a = jax.random.choice(subkey, q_values.shape[-1])
    condition = jax.random.uniform(rng) < epsilon
    action = jnp.where(condition, a, jnp.argmax(q_values, axis=-1))

    ################

    return action


def compute_loss(dqn: DQN, params: DQNParameters, target_params: DQNParameters, transition: Transition,
                 gamma: float) -> chex.Array:
    """ Computes the Deep Q-Network loss.

    Args:
        dqn (DQN): the Deep Q-Network model object
        params: (DQNParameters): the parameters of DQN
        target_params: (DQNParameters): the parameters of the target network of DQN
        transition (Transition): a tuple of (state, action, reward, done, next_state).
            shapes do not have batch dimension i.e. reward has shape (1,).
            For details on shapes, see buffers.py
            This is done for simplicity of implementation.
        gamma (float): the discounting factor for our agent
    Returns:
        loss (chex.Array): a scalar loss value
    """
    state, action, reward, done, next_state = transition

    ################
    ## YOUR CODE GOES HERE
    q_values = dqn.apply(params, state)
    next_q_values = dqn.apply(target_params, next_state)
    max_next_q_values = jnp.max(next_q_values, axis=-1)
    residual = q_values[jnp.arange(q_values.shape[0])][action] - (reward + gamma * (1 - done) * max_next_q_values)
    loss = jnp.mean(residual ** 2)

    ################
    return loss


def update_target(state: DQNTrainState) -> DQNTrainState:
    """ performs an update of the target network parameters

    Args:
        state (DQNTrainState): the current training state of DQN
    Returns:
        new_state (DQNTrainState): updated training state where target network parameters is a copy of
        the current network parameters
    """
    ################
    ## YOUR CODE GOES HERE
    new_state = state
    new_state = state.replace(target_params=state.params)

    ################

    return new_state


def initialize_agent_state(dqn: DQN, rng: chex.PRNGKey, args: DQNTrainingArgs) -> DQNTrainState:
    """ Creates the training state for the DQN agent

    Args:
        dqn (DQN): The Deep Q-Network object
        rng (chex.PRNGKey):
        args (DQNTrainingArgs): the arguments object that defines the optimization process of our agent
    Returns:
        train_state (DQNTrainState): the flax TrainingState object with an additional field
        (target network parameters) that we defined above.
    """
    ################
    ## YOUR CODE GOES HERE
    dummy_state = jnp.ones((args.train_batch_size, *dqn.state_shape))
    key, subkey = jax.random.split(rng, 2)
    p = dqn.init(subkey, dummy_state)
    target_p = p
    tx = optax.sgd(learning_rate=args.learning_rate)

    train_state = DQNTrainState.create(
        apply_fn=dqn.apply,
        params=p,
        target_params=target_p,
        tx=tx,
    )

    return train_state


# we are using cartpole dqn so we can fix the sizes
dqn = DQN(n_actions=2, state_shape=(4,))
SimpleDQNAgent = DQNAgent(
    dqn=dqn,
    initialize_agent_state=initialize_agent_state,
    select_action=select_action,
    compute_loss=compute_loss,
    update_target=update_target,
)


def compute_loss_double_dqn(dqn: DQN, params: DQNParameters, target_params: DQNParameters, transition: Transition,
                            gamma: float) -> chex.Array:
    """ Computes the Deep Q-Network loss.

    Args:
        dqn (DQN): the Deep Q-Network model object
        params: (DQNParameters): the parameters of DQN
        target_params: (DQNParameters): the parameters of the target network of DQN
        transition (Transition): a tuple of (state, action, reward, done, next_state).
            shapes do not have batch dimension i.e. reward has shape (1,).
            For details on shapes, see buffers.py
            This is done for simplicity of implementation.
        gamma (float): the discounting factor for our agent
    Returns:
        loss (chex.Array): a scalar loss value
    """
    state, action, reward, done, next_state = transition

    ################
    ## YOUR CODE GOES HERE
    q_values = dqn.apply(params, state)
    next_q_values = dqn.apply(params, next_state)
    best_action = jnp.argmax(next_q_values, axis=-1)
    target_next_q_values = dqn.apply(target_params, next_state)
    target_q_value = target_next_q_values[jnp.arange(target_next_q_values.shape[0])][best_action]
    target = reward + (1.0 - done) * gamma * target_q_value

    q_value = q_values[jnp.arange(q_values.shape[0])][action]
    loss = jnp.mean((q_value - target) ** 2)

    ################

    return loss


DoubleDQNAgent = DQNAgent(
    dqn=dqn,
    initialize_agent_state=initialize_agent_state,
    select_action=select_action,
    compute_loss=compute_loss_double_dqn,
    update_target=update_target,
)

