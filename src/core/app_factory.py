"""FastAPI app factory with Keycloak OAuth2 wired into Swagger UI."""

from fastapi import FastAPI
from fastapi.security import OAuth2AuthorizationCodeBearer

from src.core.config import config


def create_app(**kwargs) -> FastAPI:
    """Create a FastAPI app with Swagger OAuth2 configured for Keycloak.

    When Keycloak is configured, the Swagger UI will show an "Authorize"
    button that lets users log in with their Keycloak credentials directly
    in the browser — no manual token copy-paste needed.

    Args:
        **kwargs: Passed through to FastAPI constructor.

    Returns:
        Configured FastAPI instance.
    """
    swagger_ui_init_oauth = None
    openapi_extra: dict = {}

    if config.KEYCLOAK_URL and config.KEYCLOAK_REALM and config.KEYCLOAK_CLIENT_ID:
        base = config.KEYCLOAK_URL.rstrip("/")
        realm = config.KEYCLOAK_REALM
        client_id = config.KEYCLOAK_CLIENT_ID

        auth_url = f"{base}/realms/{realm}/protocol/openid-connect/auth"
        token_url = f"{base}/realms/{realm}/protocol/openid-connect/token"

        # Wire OAuth2 scheme so Swagger knows where to redirect for login
        oauth2_scheme = OAuth2AuthorizationCodeBearer(
            authorizationUrl=auth_url,
            tokenUrl=token_url,
            auto_error=False,
        )

        swagger_ui_init_oauth = {
            "clientId": client_id,
            "usePkceWithAuthorizationCodeGrant": True,
        }

        openapi_extra = {
            "components": {
                "securitySchemes": {
                    "oauth2": {
                        "type": "oauth2",
                        "flows": {
                            "authorizationCode": {
                                "authorizationUrl": auth_url,
                                "tokenUrl": token_url,
                                "scopes": {"openid": "OpenID Connect"},
                            },
                            "clientCredentials": {
                                "tokenUrl": token_url,
                                "scopes": {"openid": "OpenID Connect"},
                            },
                        },
                    }
                }
            }
        }

    app = FastAPI(
        swagger_ui_init_oauth=swagger_ui_init_oauth,
        **kwargs,
    )

    if openapi_extra:
        original_openapi = app.openapi

        def custom_openapi():
            schema = original_openapi()
            schema.setdefault("components", {}).update(
                openapi_extra.get("components", {})
            )
            return schema

        app.openapi = custom_openapi

    return app
