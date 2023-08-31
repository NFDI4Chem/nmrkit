"""create_main_tables

Revision ID: 200a761f2f51
Revises:
Create Date: 2023-07-21 09:13:50.233260

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic
revision = "200a761f2f51"
down_revision = None
branch_labels = None
depends_on = None


def create_molecular_registration_tables() -> None:
    op.create_table(
        "registration_metadata", sa.Column("key", sa.Text), sa.Column("value", sa.Text)
    )
    op.create_table(
        "hashes",
        sa.Column("molregno", sa.Integer, primary_key=True),
        sa.Column("fullhash", sa.Text, unique=True),
        sa.Column("formula", sa.Text),
        sa.Column("canonical_smiles", sa.Text),
        sa.Column("no_stereo_smiles", sa.Text),
        sa.Column("tautomer_hash", sa.Text),
        sa.Column("no_stereo_tautomer_hash", sa.Text),
        sa.Column("escape", sa.Text),
        sa.Column("sgroup_data", sa.Text),
        sa.Column("rdkitVersion", sa.Text),
    )
    op.create_table(
        "orig_data",
        sa.Column("molregno", sa.Integer, primary_key=True),
        sa.Column("data", sa.Text),
        sa.Column("datatype", sa.Text),
    )
    op.create_table(
        "molblocks",
        sa.Column("molregno", sa.Integer, primary_key=True),
        sa.Column("molblock", sa.Text),
        sa.Column("standardization", sa.Text),
    )


def upgrade() -> None:
    create_molecular_registration_tables()


def downgrade() -> None:
    op.drop_table("registration_metadata")
    op.drop_table("hashes")
    op.drop_table("orig_data")
    op.drop_table("molblocks")
