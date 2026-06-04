#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import sys

from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import (
    FlowRunFilter,
    FlowRunFilterDeploymentId,
    FlowRunFilterState,
    FlowRunFilterStateType,
)
from prefect.client.schemas.objects import StateType
from prefect.client.schemas.sorting import FlowRunSort
from prefect.states import Cancelled


ACTIVE_STATE_TYPES = (
    StateType.SCHEDULED,
    StateType.PENDING,
    StateType.RUNNING,
    StateType.CANCELLING,
    StateType.PAUSED,
)


async def cancel_deployment_runs(deployment_name: str) -> int:
    async with get_client() as client:
        deployment = await client.read_deployment_by_name(deployment_name)
        flow_runs = await client.read_flow_runs(
            flow_run_filter=FlowRunFilter(
                deployment_id=FlowRunFilterDeploymentId(any_=[deployment.id]),
                state=FlowRunFilterState(
                    type=FlowRunFilterStateType(any_=list(ACTIVE_STATE_TYPES)),
                ),
            ),
            sort=FlowRunSort.START_TIME_DESC,
            limit=100,
        )

        if not flow_runs:
            print(f"No active Prefect flow runs found for {deployment_name}")
            return 0

        print(f"Cancelling {len(flow_runs)} active Prefect flow run(s)")
        for flow_run in flow_runs:
            await client.set_flow_run_state(
                flow_run.id,
                Cancelled(message="Cancelled by pipeline restart/stop script"),
                force=True,
            )
            print(f"Cancelled {flow_run.id}")

        return len(flow_runs)


def main() -> None:
    deployment_name = sys.argv[1] if len(sys.argv) > 1 else "riot-pipeline/riot-pipeline"
    asyncio.run(cancel_deployment_runs(deployment_name))


if __name__ == "__main__":
    main()
