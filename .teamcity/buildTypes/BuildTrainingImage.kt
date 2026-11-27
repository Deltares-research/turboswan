package buildTypes

import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.dockerRegistryConnections
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.triggers.vcs

object BuildTrainingImage : BuildType({
    name = "Build training image"

    params {
        param("CONTAINER_REGISTRY", "containers.deltares.nl")
        param("CONTAINER_PROJECT", "turboswan")
        param("PACKAGE_VERSION", "unset")
        param("CONTAINER_REPO", "training")
        param("IMAGE_TAG", "unset")
        param("SKIP_BUILD", "false")
        param("env.IS_DEFAULT_BRANCH", "%teamcity.build.branch.is_default%")
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
                
                if [ "${'$'}IS_DEFAULT_BRANCH" = "true" ]; then
                    TAG="${'$'}VERSION"
                else
                    SHORT_SHA=${'$'}(echo "%build.vcs.number%" | cut -c1-7)
                    TAG="dev-${'$'}VERSION-${'$'}SHORT_SHA"
                fi
                
                echo "Image tag: ${'$'}TAG"
                echo "##teamcity[setParameter name='PACKAGE_VERSION' value='${'$'}VERSION']"
                echo "##teamcity[setParameter name='IMAGE_TAG' value='${'$'}TAG']"
            """.trimIndent()
            dockerImage = "%CONTAINER_REGISTRY%/docker-proxy/python:3.12-slim"
        }
        script {
            name = "Skip release if image already exists"
            id = "Skip_release_if_image_already_exists"
            scriptContent = """
                IMAGE="%CONTAINER_REGISTRY%/%CONTAINER_PROJECT%/%CONTAINER_REPO%:%IMAGE_TAG%"
                
                echo "Checking whether image already exists: ${'$'}IMAGE"
                
                if OUTPUT=${'$'}(docker manifest inspect "${'$'}IMAGE" 2>&1); then
                    echo "Image already exists: ${'$'}IMAGE. Skipping build and push."
                    echo "##teamcity[setParameter name='SKIP_BUILD' value='true']"
                    exit 0
                fi
                
                if echo "${'$'}OUTPUT" | grep -Eiq "no such manifest|manifest unknown|artifact .* not found"; then
                    echo "Image tag is available: ${'$'}IMAGE"
                else
                    echo "Unexpected error checking ${'$'}IMAGE:"
                    echo "${'$'}OUTPUT"
                    exit 1
                fi
            """.trimIndent()
        }
        script {
            name = "Check AI Platform Nexus credentials"
            id = "Check_AI_Platform_Nexus_credentials"
            conditions {
                doesNotEqual("SKIP_BUILD", "true")
            }
            scriptContent = """
                set +x
                python - <<'PY'
                import base64
                import hashlib
                import os
                import sys
                import urllib.error
                import urllib.request

                url = "https://internal-artifacts.deltares.nl/repository/aiplatform-python-dev/simple/"
                names = ("UV_INDEX_AIPLATFORM_PYTHON_DEV_USERNAME", "UV_INDEX_AIPLATFORM_PYTHON_DEV_PASSWORD")
                values = {name: os.environ.get(name, "") for name in names}

                unresolved = False
                for name, value in values.items():
                    digest = hashlib.sha256(value.encode()).hexdigest()[:12]
                    print(f"{name}: length={len(value)} sha256={digest}")
                    if not value:
                        print(f"  {name} is empty or unset.")
                        unresolved = True
                    elif "keeper:" in value or value.startswith("credentialsJSON:"):
                        print(f"  {name} was not resolved (still a Keeper/credentialsJSON reference).")
                        unresolved = True
                    elif value != value.strip():
                        print(f"  {name} has leading/trailing whitespace.")
                if unresolved:
                    sys.exit("AI Platform Nexus credentials were not resolved.")

                class NoRedirect(urllib.request.HTTPRedirectHandler):
                    def redirect_request(self, req, fp, code, msg, headers, newurl):
                        return None

                opener = urllib.request.build_opener(NoRedirect())

                def status(request):
                    try:
                        with opener.open(request, timeout=20) as response:
                            return response.status
                    except urllib.error.HTTPError as error:
                        return error.code
                    except urllib.error.URLError as error:
                        sys.exit(f"Nexus request failed: {error.reason}")

                print(f"Anonymous HTTP status: {status(urllib.request.Request(url))}")

                credentials = base64.b64encode(
                    f"{values[names[0]]}:{values[names[1]]}".encode()
                ).decode("ascii")
                authenticated = status(
                    urllib.request.Request(url, headers={"Authorization": f"Basic {credentials}"})
                )
                print(f"Authenticated HTTP status: {authenticated}")
                if authenticated != 200:
                    sys.exit("Credentials resolved, but Nexus rejected them.")
                print("Credentials can read the AI Platform development index.")
                PY
            """.trimIndent()
            dockerImage = "%CONTAINER_REGISTRY%/docker-proxy/python:3.12-slim"
        }
        script {
            name = "Build image"
            id = "Build_image"
            conditions {
                doesNotEqual("SKIP_BUILD", "true")
            }
            scriptContent = """
                IMAGE="%CONTAINER_REGISTRY%/%CONTAINER_PROJECT%/%CONTAINER_REPO%:%IMAGE_TAG%"
                
                echo "Building: ${'$'}IMAGE"
                
                docker build \
                    --secret id=uv_index_username,env=UV_INDEX_PYTHON_INTERNAL_USERNAME \
                    --secret id=uv_index_password,env=UV_INDEX_PYTHON_INTERNAL_PASSWORD \
                    --secret id=uv_index_aiplatform_username,env=UV_INDEX_AIPLATFORM_PYTHON_DEV_USERNAME \
                    --secret id=uv_index_aiplatform_password,env=UV_INDEX_AIPLATFORM_PYTHON_DEV_PASSWORD \
                    --tag "${'$'}IMAGE" .
            """.trimIndent()
        }
        script {
            name = "Push image"
            id = "Push_image"
            conditions {
                doesNotEqual("SKIP_BUILD", "true")
            }
            scriptContent = """
                IMAGE="%CONTAINER_REGISTRY%/%CONTAINER_PROJECT%/%CONTAINER_REPO%:%IMAGE_TAG%"
                
                echo "Pushing: ${'$'}IMAGE"
                
                docker push "${'$'}IMAGE"
            """.trimIndent()
        }
    }

    triggers {
        vcs {
            branchFilter = "+:<default>"
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
