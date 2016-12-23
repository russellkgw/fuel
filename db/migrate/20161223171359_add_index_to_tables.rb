class AddIndexToTables < ActiveRecord::Migration[5.0]
  def up
    execute <<-SQL
      CREATE INDEX exchange_rate_index ON exchange_rates (date DESC);
      CREATE INDEX oil_price_index ON oil_prices (date DESC);
      CREATE INDEX fuel_price_index ON fuel_prices (date DESC);
    SQL
  end

  def down
    execute <<-SQL
      DROP INDEX IF EXISTS exchange_rate_index;
      DROP INDEX IF EXISTS oil_price_index;
      DROP INDEX IF EXISTS fuel_price_index;
    SQL
  end
end
