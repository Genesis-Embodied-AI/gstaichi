from pydantic import BaseModel

from gstaichi.lang._wrap_inspect import FunctionSourceInfo


class HashedFunctionSourceInfo(BaseModel):
    function_source_info: FunctionSourceInfo
    hash: str
