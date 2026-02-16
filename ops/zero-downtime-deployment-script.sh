    #!/bin/bash

    set -e

    # Define variables
    PROJECT_DIR="/mnt/data/nmrkit"
    COMPOSE_FILE="docker-compose-prod.yml"
    NMRKIT_IMAGE="nfdi4chem/nmrkit:dev-latest"
    NMR_CLI_IMAGE="nfdi4chem/nmr-cli:dev-latest"
    NEW_CONTAINER_ID=""
    LOG_FILE="/var/log/nmrkit-deploy.log"

    # Create log file if it doesn't exist
    if [ ! -f "$LOG_FILE" ]; then
        sudo touch "$LOG_FILE"
        sudo chmod 644 "$LOG_FILE"
    fi

    # Unified logging function
    log_message() {
        echo "$1"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
    }

    # === Start of script ===
    log_message "🚀 =========================================="
    log_message "🚀 Starting NMRKit Deployment Script"
    log_message "🚀 =========================================="
    
    # Change to project directory to ensure paths resolve correctly
    cd "$PROJECT_DIR/ops" || {
        log_message "❌ Failed to change to directory $PROJECT_DIR/ops"
        exit 1
    }
    log_message "📂 Working directory: $(pwd)"

    # === Functions ===

    # Function to check the health of the container
    check_health() {
        if ! HEALTH=$(docker inspect --format='{{json .State.Health.Status}}' "$NEW_CONTAINER_ID" 2>/dev/null); then
            log_message "❌ Failed to inspect container $NEW_CONTAINER_ID. Ensure the container ID is correct."
            return 1
        fi
        
        log_message "Health status for $NEW_CONTAINER_ID: $HEALTH"
        
        if [[ "$HEALTH" == *"healthy"* ]]; then
            log_message "✅ Container is healthy."
            return 0
        else
            log_message "❌ Container $NEW_CONTAINER_ID is not healthy."
            return 1
        fi
    }

    # Wait for container to pass health check
    wait_for_health() {
        log_message "⏳ Waiting for new container to pass health check (up to 10 retries)..."
        
        for i in {1..10}; do
            if check_health; then
                return 0
            else
                log_message "Retry $i/10: Waiting 60s..."
                sleep 60
            fi
        done
        
        log_message "❌ Container health check failed after 10 retries."
        return 1
    }

    # Cleanup function
    cleanup() {
        log_message "🧼 Cleaning up unused containers and images..."
        docker container prune -f >/dev/null 2>&1 || true
        docker image prune -f >/dev/null 2>&1 || true
        log_message "✅ Cleanup completed"
    }

    # Remove old containers
    remove_old_containers() {
        local container_prefix=$1
        log_message "🗑️  Removing old ${container_prefix} container(s)..."
        
        # Retrieve the container IDs that match the prefix
        container_ids=$(docker ps -a --filter "name=${container_prefix}" --format "{{.ID}}")
        
        if [ -z "$container_ids" ]; then
            log_message "❌ No containers found with name prefix: ${container_prefix}"
            return 1
        fi
        
        # Sort the container IDs by creation date in ascending order
        sorted_container_ids=$(echo "$container_ids" | xargs docker inspect --format='{{.Created}} {{.ID}}' | sort | awk '{print $2}')
        
        # Get the oldest container ID
        oldest_container_id=$(echo "$sorted_container_ids" | head -n 1)
        
        if [ -z "$oldest_container_id" ]; then
            log_message "❌ No old container found to remove"
            return 1
        fi
        
        log_message "Stopping container: $oldest_container_id"
        docker stop "$oldest_container_id"
        docker rm "$oldest_container_id"
        log_message "✅ Deleted old container ID: $oldest_container_id"
    }

    # Deploy a service with zero downtime
    deploy_service() {
        local service_name=$1
        local image=$2
        local scale_count=${3:-2}
        
        log_message "📦 Starting deployment for service: $service_name"
        log_message "🔍 Checking for new image: $image"
        
        # Pull the latest image
        if [ "$(docker pull "$image" | grep -c "Status: Image is up to date")" -eq 0 ]; then
            log_message "✨ New image available for $service_name"
            
            # Scale up a new container
            log_message "⚡ Scaling up new container for $service_name..."
            docker-compose -f "$COMPOSE_FILE" up -d --scale "$service_name=$scale_count" --no-recreate
            
            # Get the new container ID
            NEW_CONTAINER_ID=$(docker ps -q -l)
            log_message "🆕 New container ID: $NEW_CONTAINER_ID"
            
            # Wait for the new container to be healthy (only if service has healthcheck)
            if [[ "$service_name" == "nmrkit-api" ]]; then
                if wait_for_health; then
                    log_message "✅ New container is healthy, proceeding with old container removal"
                    
                    # Remove old containers
                    if remove_old_containers "$service_name"; then
                        cleanup
                        log_message "✅ Deployment of $service_name completed successfully"
                    else
                        log_message "⚠️  Could not remove old containers, but new container is running"
                    fi
                else
                    log_message "❌ Deployment aborted: new $service_name container is unhealthy"
                    log_message "🔄 Rolling back: Stopping and removing unhealthy container"
                    docker stop "$NEW_CONTAINER_ID"
                    docker rm "$NEW_CONTAINER_ID"
                    return 1
                fi
            else
                # For services without healthcheck, wait a bit and then remove old containers
                log_message "⏳ Waiting 30s for service to stabilize..."
                sleep 30
                
                if remove_old_containers "$service_name"; then
                    cleanup
                    log_message "✅ Deployment of $service_name completed successfully"
                else
                    log_message "⚠️  Could not remove old containers, but new container is running"
                fi
            fi
        else
            log_message "✅ No update for $service_name. Skipping deployment."
        fi
    }

    # Main deployment process
    main() {
        log_message "────────────────────────────────────────"
        log_message "🔄 Deploying NMRKit API Service"
        log_message "────────────────────────────────────────"
        deploy_service "nmrkit-api" "$NMRKIT_IMAGE"
        
        log_message ""
        log_message "────────────────────────────────────────"
        log_message "🔄 Deploying NMR-Load-Save Service"
        log_message "────────────────────────────────────────"
        deploy_service "nmr-load-save" "$NMR_CLI_IMAGE"
        
        log_message ""
        log_message "🎉 =========================================="
        log_message "🎉 All Deployments Completed Successfully!"
        log_message "🎉 =========================================="
    }

    # Execute main deployment
    main
