from typing import Optional

from pytorch_lightning.callbacks import Callback


class RegularizationAnnealing(Callback):
    """
    General callback for annealing loss scales over training.
    
    :param loss_name: Loss name to anneal the scale for.
    :param schedule: Annealing schedule type ('linear' or 'cyclic').
    :param start_value: Initial scale value.
    :param end_value: Final scale value.
    :param start_epoch: Epoch to start annealing.
    :param end_epoch: Epoch to end annealing.
    :param cycle_length: Length of each cycle in epochs (if cyclic is True).
    """
    def __init__(
        self,
        loss_name: str,
        schedule: Optional[str] = "linear",
        start_value: Optional[float] = 0.0,
        end_value: Optional[float] = 1.0,
        start_epoch: Optional[int] = 0,
        end_epoch: Optional[int] = 50,
        cycle_length: Optional[int] = 100,
    ):
        super().__init__()
        self.loss_name = loss_name
        self.schedule = schedule
        self.start_value = start_value
        self.end_value = end_value
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.cycle_length = cycle_length
        self.log_key = "train/annealing_scale"

    def _cyclic_annealing(self, epoch: int) -> float:
        """Cyclic annealing schedule."""
        cycle_pos = (epoch % self.cycle_length) / max(1, self.cycle_length)
        return self.start_value + (self.end_value - self.start_value) * cycle_pos
    
    def _linear_annealing(self, epoch: int) -> float:
        """Linear annealing schedule."""
        if epoch < self.start_epoch:
            return self.start_value
        elif epoch > self.end_epoch:
            return self.end_value

        progress = (epoch - self.start_epoch) / max(1, self.end_epoch - self.start_epoch)
        return self.start_value + progress * (self.end_value - self.start_value)

    def anneal(self, epoch: int) -> float:
        """Compute the annealed value for a given epoch."""
        if self.schedule == "cyclic":
            return self._cyclic_annealing(epoch)
        return self._linear_annealing(epoch)
    
    def _update_scale(self, pl_module, scale: float):
        """Update the scale of the targeted losses in the model."""

        # If there are only two losses, preserve convex combination
        if hasattr(pl_module.loss, "losses") and len(pl_module.loss.list_losses) == 2:
            loss_name0, loss_name1 = pl_module.loss.list_losses
            if loss_name0 == self.loss_name:
                getattr(pl_module.loss.losses, loss_name0).scale = scale
                getattr(pl_module.loss.losses, loss_name1).scale = 1.0 - scale
            elif loss_name1 == self.loss_name:
                getattr(pl_module.loss.losses, loss_name1).scale = scale
                getattr(pl_module.loss.losses, loss_name0).scale = 1.0 - scale
            else:
                # fallback: just update the target loss
                getattr(pl_module.loss.losses, self.loss_name).scale = scale

        # Otherwise just update the target loss
        else:
            getattr(pl_module.loss.losses, self.loss_name).scale = scale

    def on_train_epoch_start(self, trainer, pl_module):
        """Update loss scales at the beginning of each epoch."""
        
        # Get the new annealed scale and update losses:
        scale = self.anneal(trainer.current_epoch)
        self._update_scale(pl_module, scale)
        
        # Log the annealing coefficient
        pl_module.log_dict(
            {self.log_key: scale},
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )


class InformationBottleneckAnnealing(RegularizationAnnealing):
    """
    Information Bottleneck-oriented annealing for a VAE-like setup.

    Strategies:
      - strategy='plain': classic warmup/ramp of alpha (low complexity)
      - strategy='capacity': track a target KL capacity C_t and adjust alpha to match
      - strategy='pid': PID controller on (KL - C_t) for smoother/stabler control

    :param loss_name: Loss name to anneal the scale for (typically 'kl').
    :param schedule: Annealing schedule type ('linear' or 'cyclic').
    :param start_epoch: Epoch to start annealing.
    :param end_epoch: Epoch to end annealing.
    :param alpha_max: Maximum value of the annealing coefficient alpha.
    :param alpha_min: Minimum value of the annealing coefficient alpha.
    :param cycle_length: Length of each cycle in epochs (only for cyclic).
    :param warmup_ratio: Fraction of each cycle with alpha=0 (only for cyclic).
    :param strategy: IB strategy ('plain', 'capacity', or 'pid').
    :param cap_gain: Proportional gain for capacity control (only for 'capacity' strategy).
    :param cap_start_epoch: Epoch to start capacity schedule (default: start_epoch).
    :param cap_end_epoch: Epoch to end capacity schedule (default: end_epoch).
    :param cap_start: Initial capacity C_0.
    :param cap_max: Maximum capacity C_T (nats).
    :param cap_cyclic: Whether to make capacity cyclic (only for cyclic schedule).
    :param pid_kp: Proportional gain for PID controller (only for 'pid' strategy).
    :param pid_ki: Integral gain for PID controller (only for 'pid' strategy).
    :param pid_kd: Derivative gain for PID controller (only for 'pid' strategy).
    :param pid_integral_clip: Clipping value for PID integral term (only for 'pid' strategy).
    :param kl_metric_key: Key to read the KL metric from callback_metrics.
    """

    def __init__(
        self,
        loss_name: str = "kl",
        schedule: Optional[str] = "linear",
        start_epoch: int = 0,
        end_epoch: int = 50,
        alpha_max: float = 1.0,
        alpha_min: float = 0.0,
        cycle_length: int = 100, 
        warmup_ratio: float = 0.2,
        strategy: str = "plain",

        # KL capacity schedule C_t (used by 'capacity' and 'pid')
        cap_gain: float = 0.02,
        cap_start_epoch: Optional[int] = None,
        cap_end_epoch: Optional[int] = None,
        cap_start: float = 0.0,
        cap_max: float = 20.0,
        cap_cyclic: bool = False,

        # PID controller knobs (used by 'pid')
        pid_kp: float = 0.02,
        pid_ki: float = 0.0,
        pid_kd: float = 0.0,
        pid_integral_clip: float = 5.0,

        # Where to read the KL metric from callback_metrics
        kl_metric_key: str = "train/loss/kl",
    ):
        super().__init__(
            loss_name=loss_name,
            schedule=schedule,
            start_value=alpha_min,
            end_value=alpha_max,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            cycle_length=cycle_length,
        )
        self.log_key = "train/alpha"
        # IB hyperparams
        self.alpha_max, self.alpha_min = alpha_max, alpha_min
        self.warmup_ratio = max(0.0, min(0.95, warmup_ratio))
        self.strategy = strategy

        # Capacity schedule
        self.cap_gain = cap_gain
        self.cap_start_epoch = start_epoch if cap_start_epoch is None else cap_start_epoch
        self.cap_end_epoch = end_epoch if cap_end_epoch is None else cap_end_epoch
        self.cap_start = cap_start
        self.cap_max = cap_max
        self.cap_cyclic = cap_cyclic

        # PID
        self.pid_kp = pid_kp
        self.pid_ki = pid_ki
        self.pid_kd = pid_kd
        self.pid_integral_clip = abs(pid_integral_clip)

        # Metric
        self.kl_metric_key = kl_metric_key
        
        # State updated across epochs
        self._last_alpha = 0.0
        self._last_kl = None
        self._pid_integral = 0.0
        self._pid_prev_error = None


    def on_train_epoch_end(self, trainer, pl_module):
        """Capture the latest KL metric for use in the next epoch's alpha."""
        cm = getattr(trainer, "callback_metrics", {}) or {}
        self._last_kl = cm.get(self.kl_metric_key, self._last_kl)

    def _capacity_linear(self, epoch: int) -> float:
        """Monotonic linear capacity schedule C_t between cap_start_epoch and cap_end_epoch."""
        if epoch <= self.cap_start_epoch:
            return self.cap_start
        if epoch >= self.cap_end_epoch:
            return self.cap_max
        progress = (epoch - self.cap_start_epoch) / max(1, (self.cap_end_epoch - self.cap_start_epoch))
        return self.cap_start + progress * (self.cap_max - self.cap_start)

    def _capacity_cyclic(self, epoch: int) -> float:
        """Cyclic capacity schedule: warmup to cap_max within the active part of each cycle."""
        pos = (epoch % self.cycle_length) / max(1, self.cycle_length)  # [0,1)
        if pos < self.warmup_ratio:
            return self.cap_start
        ramp_len = max(1e-6, 1.0 - self.warmup_ratio)
        progress = (pos - self.warmup_ratio) / ramp_len  # 0→1
        return self.cap_start + progress * (self.cap_max - self.cap_start)

    def _capacity(self, epoch: int, cyclic: bool) -> float:
        return self._capacity_cyclic(epoch) if (cyclic and self.cap_cyclic) else self._capacity_linear(epoch)

    def _apply_pid(self, target: float, observed: Optional[float]) -> float:
        """
        PID controller that adjusts alpha based on error = observed_kl - target_capacity.
        If observed is None (not yet logged), we keep alpha as-is.
        """
        alpha = self._last_alpha
        if observed is None:
            return alpha

        error = float(observed) - float(target)

        # Integral (clipped)
        self._pid_integral += error
        self._pid_integral = max(-self.pid_integral_clip, min(self.pid_integral_clip, self._pid_integral))

        # Derivative
        derivative = 0.0 if self._pid_prev_error is None else (error - self._pid_prev_error)

        delta = self.pid_kp * error + self.pid_ki * self._pid_integral + self.pid_kd * derivative
        self._pid_prev_error = error

        new_alpha = alpha + delta
        return max(self.alpha_min, min(self.alpha_max, new_alpha))

    def _linear_annealing(self, epoch: int) -> float:
        """
        Strategy-specific linear scheduler.
        - plain: 0 → alpha_max linearly from start_epoch to end_epoch
        - capacity: adjust alpha to drive KL towards C_t (monotonic capacity)
        - pid: PID controller on (KL - C_t) with monotonic capacity
        """            

        if self.strategy == "capacity":
            # Compute a linear capacity target and do a simple proportional update of alpha
            C_t = self._capacity(epoch, cyclic=False)

            # Simple proportional controller on error (KL - C_t)
            if self._last_kl is None:
                alpha = super()._linear_annealing(epoch)
            else:
                alpha = self._last_alpha + self.cap_gain * (float(self._last_kl) - float(C_t))
                alpha = max(self.alpha_min, min(self.alpha_max, alpha))

        elif self.strategy == "pid":
            # PID on error = KL - C_t
            C_t = self._capacity(epoch, cyclic=False)
            alpha = self._apply_pid(target=C_t, observed=self._last_kl)

        else:
            return super()._linear_annealing(epoch)

        self._last_alpha = alpha
        return alpha

    def _plain_cyclic_annealing(self, epoch: int) -> float:
        """Cyclic annealing schedule."""
        cycle_pos = (epoch % self.cycle_length) / max(1, self.cycle_length)
        if cycle_pos < self.warmup_ratio:
            alpha = self.start_value
        else:
            ramp_len = max(1e-6, 1.0 - self.warmup_ratio)
            progress = (cycle_pos - self.warmup_ratio) / ramp_len  # 0→1
            alpha = self.alpha_max * max(0.0, min(1.0, progress))
        return alpha

    def _cyclic_annealing(self, epoch: int) -> float:
        """
        Strategy-specific cyclic scheduler.
        - plain: per-cycle warmup (alpha=0) for warmup_ratio, then linear ramp to alpha_max
        - capacity: define a (possibly cyclic) capacity C_t; simple proportional update of alpha
        - pid: PID on (KL - C_t) with cyclic C_t
        """ 

        if self.strategy == "capacity":
            C_t = self._capacity(epoch, cyclic=True)
            if self._last_kl is None:
                alpha = self._plain_cyclic_annealing(epoch)
            else:
                alpha = self._last_alpha + self.cap_gain * (float(self._last_kl) - float(C_t))
                alpha = max(self.alpha_min, min(self.alpha_max, alpha))

        elif self.strategy == "pid":
            C_t = self._capacity(epoch, cyclic=True)
            alpha = self._apply_pid(target=C_t, observed=self._last_kl)

        else:
            alpha = self._plain_cyclic_annealing(epoch)

        self._last_alpha = alpha
        return alpha
