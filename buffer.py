import jax 
import chex
from jax import numpy as jnp

from typing import Tuple, Any, Callable


@chex.dataclass(frozen=True)
class ReplayBufferStorage:
  states: chex.Array # dtype float32, shape [buffer_size, *state_shape]
  actions: chex.Array # dtype int32, shape [buffer_size, 1]
  rewards: chex.Array # dtype float32, shape [buffer_size, 1]
  dones: chex.Array # dtype bool, shape [buffer_size, 1]
  next_states: chex.Array # dtype float32, shape [buffer_size, *state_shape]
  cursor: chex.Array # dtype int32, shape [1] - the pointer right after the most recently added element
  full: chex.Array # dtype bool, shape [1] - boolean flag indicating whether the current buffer is full

Transition = chex.ArrayTree # - Transition is a tuple of (state, action, reward, done, next_state)


@chex.dataclass(frozen=True)
class ReplayBuffer:
  init_buffer: Callable[[Any], ReplayBufferStorage]
  """init_buffer: initializes the replay buffer returning an empty one. It may accept any args"""
  add_transition: Callable[[ReplayBufferStorage, Transition], ReplayBufferStorage]
  """add_transition: 
    adds one transition (state, action, reward, done, next_state) to the replay buffer.
    Transition should be a tuple with arrays, each of the corresponding shape expected
    by the storage without the buffer_size dimension. For example actions should be of shape
    [*action_shape]
  """
  sample_transition: Callable[[chex.PRNGKey, ReplayBufferStorage], Transition]
  """sample_transition:
    samples ONE transition from the replay buffer. the format of transition is:
    a tuple of (state, action, reward, done, next_state).
    This function accepts random key of jax to perform random sampling
  """


def init_buffer(buffer_size: int, state_shape: Tuple[int]) -> ReplayBufferStorage:
  """ initializes an empty buffer.

  Args:
    buffer_size: int
    state_shape: Tuple[int]
  Returns:
    ReplayBufferStorage
  """
  return ReplayBufferStorage(
    states=jnp.zeros((buffer_size, *state_shape), dtype=jnp.float32),
    actions=jnp.zeros((buffer_size, 1), dtype=jnp.int32),
    rewards=jnp.zeros((buffer_size, 1), dtype=jnp.float32),
    dones=jnp.zeros((buffer_size, 1), dtype=jnp.bool_),
    next_states=jnp.zeros((buffer_size, *state_shape), dtype=jnp.float32),
    cursor=jnp.array(0), # since the buffer is empty, the cursor points at zero
    full=jnp.array(False)
  )


def add_transition(buffer: ReplayBufferStorage, transition: Transition) -> ReplayBufferStorage:
  """ adds one transition to the replay buffer.

  The implementation should follow the standard circular array pattern, i.e.,
  each new transition should be written right after (wrt the 0-th dimension) 
  the most recently added one (while the initial element should be added at index 0).
  Once the maximal size is reached, the buffer should start overwriting its previous values
  starting from the oldest one.

  Args:
    buffer (ReplayBufferStorage): the buffer storage instance
    transition (Transition): a tuple of (state, action, reward, done, next_state)
      see the definition of ReplayBufferStorage and Transition for details on shapes.
      (they don't have the leading ``batch'' dimension, e.g. reward has space (1,))
  Returns: 
    ReplayBufferStorage: an updated buffer storage instance 
    (do not forget to update buffer.cursor and buffer.full !!!)
  """
  state, action, reward, done, next_state = transition
  cursor = buffer.cursor
  max_buffer_size = buffer.rewards.shape[0]

  ################
  ## YOUR CODE GOES HERE

  new_states = buffer.states.at[cursor].set(state)
  new_actions = buffer.actions.at[cursor].set(action)
  new_rewards = buffer.rewards.at[cursor].set(reward)
  new_dones = buffer.dones.at[cursor].set(done)
  new_next_states = buffer.next_states.at[cursor].set(next_state)

  new_cursor = (cursor + 1) % max_buffer_size
  new_full = jnp.where(new_cursor == 0, True, buffer.full)

  return ReplayBufferStorage( states=new_states, actions=new_actions, rewards=new_rewards, dones=new_dones,
                               next_states=new_next_states, cursor=jnp.array(new_cursor), full=new_full)

  ################


def sample_transition(rng: chex.PRNGKey, buffer: ReplayBufferStorage) -> Transition:
  """ randomly (with uniform distribution) samples one transition to retrieve from the replay buffer.

  Args:
    rng (chex.PRNGKey): - random generation key for jax
    buffer (ReplayBufferStorage):
  Returns:
    Transition: a tuple of (state, action, reward, done, next_state)
    You should assume that rng is a "single" key, i.e. as if it was
    returned by `jax.random.key(42)` and implement this function accordingly.
    The shapes of each element of this objects should be the same as the ones 
    expected by `add_transition`.
  """

  ################
  ## YOUR CODE GOES HERE
  # please define these variables yourself
  max_size = buffer.rewards.shape[0]
  sample_size = jnp.where(buffer.full, max_size, buffer.cursor)

  index = jax.random.randint(rng, shape=(), minval=0, maxval=sample_size)

  sampled_states = buffer.states[index]
  sampled_actions = buffer.actions[index]
  sampled_rewards = buffer.rewards[index]
  sampled_dones = buffer.dones[index]
  samped_next_states = buffer.next_states[index]
  ################
  
  transition = (
    sampled_states,
    sampled_actions,
    sampled_rewards,
    sampled_dones,
    samped_next_states,
  )
  return transition

FIFOBuffer = ReplayBuffer(
    init_buffer=jax.jit(init_buffer, static_argnums=(0, 1)),
    add_transition=jax.jit(add_transition),
    sample_transition=jax.jit(sample_transition)
)