# Objective

We want to iteratively collect the elite and sub-elite players that we are comfortable with, currently that is players which fall under the following requirements:

- Queue: FlexQ and SoloQ
- Tier: Diamond
- Division: 4

This is to give us the minimum required threshold for all regions which acts as the point where we would never want to go any lower, however we would still be happy holding onto the extra data in-case we needed a higher level.

Note all games will have further processing to remove games not representative of champion quality.

## Step 1

We need to extract all of the players at the required level and above and store them in a database.

## Commands

- celery -A app.workers.app.celery_app worker --pool=solo -l info
- celery -A app.workers.app.celery_app beat -l info
- celery -A app.workers.app.celery_app call pipelines.player_collection
