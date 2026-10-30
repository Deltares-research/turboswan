package buildTypes

import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.dockerRegistryConnections
import jetbrains.buildServer.configs.kotlin.buildSteps.script

object BuildTrainingImage : BuildType({
    name = "Build training image"

    params {
        param("CONTAINER_REGISTRY", "containers.deltares.nl")
        param("CONTAINER_PROJECT", "turboswan")
        param("PACKAGE_VERSION", "unset")
        param("CONTAINER_REPO", "training")
    }

    vcs {
        root(DslContext.settingsRoot)
    }

    steps {
        script {
            name = "Read version"
            id = "Read_version"
            scriptContent = """
                VERSION=${'$'}(python3 -c "
                import tomllib
                
                with open('pyproject.toml', 'rb') as f:
                    print(tomllib.load(f)['project']['version'])
                ")
                
                echo "Detected package version: ${'$'}VERSION"
                echo "##teamcity[setParameter name='PACKAGE_VERSION' value='${'$'}VERSION']"
            """.trimIndent()
            dockerImage = "containers.deltares.nl/docker-proxy/python:3.12-slim"
        }
        script {
            name = "Fail if image already exists"
            id = "Fail_if_image_already_exists"
            scriptContent = """
                IMAGE="%CONTAINER_REGISTRY%/%CONTAINER_PROJECT%/%CONTAINER_REPO%:%PACKAGE_VERSION%"
                
                echo "Checking whether image already exists: ${'$'}IMAGE"
                
                if docker manifest inspect "${'$'}IMAGE" >/dev/null 2>&1; then
                    echo "Image already exists: ${'$'}IMAGE"
                    exit 1
                fi
                
                echo "Image tag is available: ${'$'}IMAGE"
            """.trimIndent()
        }
        script {
            name = "Build image"
            id = "Build_image"
            scriptContent = """
                IMAGE="%CONTAINER_REGISTRY%/%CONTAINER_PROJECT%/%CONTAINER_REPO%:%PACKAGE_VERSION%"
                
                echo "Building: ${'$'}IMAGE"
                
                docker build \
                    --secret id=uv_index_username,env=UV_INDEX_PYTHON_INTERNAL_USERNAME \
                    --secret id=uv_index_password,env=UV_INDEX_PYTHON_INTERNAL_PASSWORD \
                    --tag "${'$'}IMAGE" .
            """.trimIndent()
        }
        script {
            name = "Push image"
            id = "Push_image"
            scriptContent = """
                IMAGE="%CONTAINER_REGISTRY%/%CONTAINER_PROJECT%/%CONTAINER_REPO%:%PACKAGE_VERSION%"
                
                echo "Pushing: ${'$'}IMAGE"
                
                docker push "${'$'}IMAGE"
            """.trimIndent()
        }
    }

    features {
        dockerRegistryConnections {
            loginToRegistry = on {
                dockerRegistryId = "PROJECT_EXT_92"
            }
        }
    }

    requirements {
        contains("teamcity.agent.jvm.os.name", "Linux")
    }
})
