class DataTables < ActiveRecord::Migration[5.0]
  def change
    create_table(:exchange_rates) do |t|
      t.string :base, limit: 255
      t.string :currency, limit: 255
      t.decimal :value
      t.date :date
      t.string :source
      t.timestamps
    end

    create_table(:oil_prices) do |t|
      t.string :currency, limit: 255
      t.decimal :value
      t.date :date
      t.string :source
      t.timestamps
    end

    create_table(:fuel_prices) do |t|
      t.string :currency, limit: 255
      t.decimal :base
      t.decimal :full
      t.date :date
      t.string :source
      t.timestamps
    end

  end
end
