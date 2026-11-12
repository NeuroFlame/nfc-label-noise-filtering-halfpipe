"""Load and validate one site's inputs for the first computation round."""

import os
import uuid

from .halfpipe_inputs import (
    STAGED_DIR_NAME,
    HalfpipeInputError,
    is_nested_halfpipe_dataset,
    stage_flat_dataset,
)
from .types import ValidatedInputs
from .validation import validate_and_get_inputs


def load_inputs(data_dir, parameters, logger, cache_dir, output_dir):
    """Validate local inputs and build the payload for the first local step.

    Nested HALFpipe derivatives are first converted into the flat per-subject
    layout in this run's cache directory (see ``halfpipe_inputs``); flat inputs
    are read as they are.

    Also mints a random ``self_token``: the framework does not expose this
    site's own identity to author code, so the token is how this site later
    recognizes which entry in the aggregator's site-keyed results is its own.
    See local_math.py's ``compute_dimensional_scores``.
    """
    if is_nested_halfpipe_dataset(data_dir, parameters):
        try:
            data_dir, parameters = stage_flat_dataset(
                data_dir,
                parameters,
                os.path.join(cache_dir, STAGED_DIR_NAME),
                logger,
                summary_dir=output_dir,
            )
        except HalfpipeInputError as error:
            logger.error(f"{error}\n\n")
            raise ValueError(
                "Invalid run input; see the site log for validation details"
            ) from error

    is_valid, data = validate_and_get_inputs(data_dir, parameters, logger)
    if not is_valid:
        raise ValueError("Invalid run input; see the site log for validation details")

    return ValidatedInputs(data=data, self_token=uuid.uuid4().hex)
