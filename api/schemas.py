from pydantic import BaseModel, Field
from typing import Optional

class TextoRequest(BaseModel):
    texto: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Texto de la PQRS a clasificar",
        examples=["Llevo 3 meses esperando respuesta a mi solicitud sin resultado."],
    )

    class Config:
        json_schema_extra = {
            "example": {
                "texto": "Llevo 3 meses esperando respuesta a mi solicitud sin resultado."
            }
        }


class BatchRequest(BaseModel):
    textos: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Lista de textos a clasificar (máximo 50)",
        examples=[["Solicito información sobre mi trámite.", "Exijo reembolso urgente."]],
    )


class EventoRequest(BaseModel):
    titulo: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Título del evento institucional",
    )
    descripcion: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Descripción detallada del evento",
    )
    area: str = Field(
        default="General",
        description="Área afectada por el evento",
    )
    vigencia_valor: int = Field(
        ...,
        ge=1,
        le=365,
        description="Duración de la vigencia del evento",
    )
    vigencia_tipo: str = Field(
        ...,
        pattern="^(horas|dias)$",
        description="Unidad de vigencia: 'horas' o 'dias'",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "titulo": "Caída del sistema de pagos",
                "descripcion": "El sistema presentó fallas entre las 10am y 2pm del 20 de mayo.",
                "area": "Pagos",
                "vigencia_valor": 3,
                "vigencia_tipo": "dias",
            }
        }

class ClasificacionResponse(BaseModel):
    texto: str
    categoria: str = Field(description="Petición | Queja | Reclamo | Sugerencia")
    urgencia: str = Field(description="Alta | Media | Baja")
    confianza: Optional[float] = Field(description="Confianza del modelo entre 0 y 100")
    probabilidades: Optional[dict[str, float]] = Field(
        description="Probabilidad por cada categoría"
    )
    nivel_riesgo: str = Field(
        default="Verde",
        description="Semáforo de riesgo: Verde | Amarillo | Rojo",
    )
    score_riesgo: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Score de riesgo (0-100) que determina el semáforo",
    )
    respuesta_sugerida: Optional[str] = Field(
        default=None,
        description="Respuesta institucional sugerida para la PQRS",
    )
    fuente_respuesta: Optional[str] = Field(
        default=None,
        description="Origen de la respuesta: plantilla o IA",
    )
    eventos_considerados: int = Field(
        default=0,
        description="Cantidad de eventos institucionales activos al momento de clasificar",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "texto": "Llevo 3 meses esperando respuesta.",
                "categoria": "Reclamo",
                "urgencia": "Alta",
                "confianza": 94.2,
                "probabilidades": {
                    "Petición": 2.1,
                    "Queja": 3.0,
                    "Reclamo": 94.2,
                    "Sugerencia": 0.7,
                },
                "nivel_riesgo": "Rojo",
                "score_riesgo": 75,
            }
        }


class BatchResponse(BaseModel):
    total:       int
    resultados:  list[ClasificacionResponse]
    resumen:     dict[str, int] = Field(description="Conteo por categoría")


class EventoResponse(BaseModel):
    id:              int
    titulo:          str
    descripcion:     str
    area:            str
    vigencia_valor:  int
    vigencia_tipo:   str
    creado_en:       str
    expira_en:       str
    activo:          bool
    tiempo_restante: str


class HealthResponse(BaseModel):
    estado:          str
    modelo_cargado:  bool
    version_api:     str
    eventos_activos: int


class InfoResponse(BaseModel):
    nombre:      str
    descripcion: str
    version:     str
    endpoints:   list[str]