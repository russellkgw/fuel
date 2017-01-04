class UpdateFuelPrices < ActiveRecord::Migration[5.0]
  def change
    remove_column :fuel_prices, :currency

    rename_column :fuel_prices, :full, :full_95_coast
    rename_column :fuel_prices, :base, :basic_fuel_price

    add_column :fuel_prices, :exchange_rate, :decimal
    add_column :fuel_prices, :crude_oil, :decimal

    add_column :fuel_prices, :bfp, :decimal
    add_column :fuel_prices, :fuel_tax, :decimal
    add_column :fuel_prices, :customs_excise, :decimal
    add_column :fuel_prices, :equalization_fund_levy, :decimal
    add_column :fuel_prices, :raf, :decimal
    add_column :fuel_prices, :transport_cost, :decimal
    add_column :fuel_prices, :petroleum_products_levy, :decimal
    add_column :fuel_prices, :wholesale_margin, :decimal
    add_column :fuel_prices, :secondary_storage, :decimal
    add_column :fuel_prices, :secondary_distribution, :decimal
    add_column :fuel_prices, :retail_margin, :decimal
    add_column :fuel_prices, :slate_levy, :decimal
    add_column :fuel_prices, :delivery_cost, :decimal
    add_column :fuel_prices, :dsml, :decimal
  end
end
